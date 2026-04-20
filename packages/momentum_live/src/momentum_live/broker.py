"""Alpaca Broker API integration for per-pair sandbox sub-accounts.

The Broker API (sandbox: ``https://broker-api.sandbox.alpaca.markets``) lets us
create one isolated sub-account per crypto pair, each with its own balance and
positions. The trading agent then sees an observation that exactly matches the
single-asset distribution it was trained on, with no contested cash pool.

This module exposes:

- :class:`BrokerCredentials` - reads ``ALPACA_BROKER_*`` env vars.
- :class:`BrokerAccountManager` - thin wrapper over ``alpaca.broker.client.BrokerClient``
  that knows how to create sub-accounts (idempotent via the registry), JNLC-fund
  them from a designated firm/funding account, and reset them back to a target
  balance between checkpoints.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from momentum_core.logging import get_logger

from .account_registry import BrokerAccountRegistry, SubAccountEntry

LOGGER = get_logger("momentum_live.broker")


@dataclass(slots=True)
class BrokerCredentials:
    """Sandbox Broker API credentials + (optional) firm/funding account id.

    ``funding_account_id`` is required for the JNLC-based reset flow (``soft``
    or ``hard``). It is optional when the runner is started with
    ``--reset-mode none`` (typical for "adopt an existing end-user sub-account
    and just validate the agent" flows).
    """

    api_key: str
    secret_key: str
    base_url: str
    funding_account_id: str = ""

    @classmethod
    def from_environment(
        cls,
        *,
        api_key_var: str = "ALPACA_BROKER_API_KEY",
        secret_key_var: str = "ALPACA_BROKER_API_SECRET",
        base_url_var: str = "ALPACA_BROKER_BASE_URL",
        funding_account_var: str = "ALPACA_BROKER_ACCOUNT_ID",
    ) -> BrokerCredentials:
        api_key = os.getenv(api_key_var)
        secret_key = os.getenv(secret_key_var)
        base_url = os.getenv(base_url_var, "https://broker-api.sandbox.alpaca.markets")
        funding_account_id = os.getenv(funding_account_var, "") or ""

        missing = [
            name
            for name, value in (
                (api_key_var, api_key),
                (secret_key_var, secret_key),
            )
            if not value
        ]
        if missing:
            raise OSError(f"Missing Broker API env vars: {', '.join(missing)}. Run `source scripts/env-paper.sh` after populating .env.")

        return cls(
            api_key=api_key,  # type: ignore[arg-type]
            secret_key=secret_key,  # type: ignore[arg-type]
            base_url=base_url,
            funding_account_id=funding_account_id,
        )

    @property
    def is_sandbox(self) -> bool:
        return "sandbox" in self.base_url.lower()

    @property
    def has_firm_account(self) -> bool:
        return bool(self.funding_account_id)


class BrokerAccountManager:
    """High-level operations on Broker sub-accounts.

    Wraps :class:`alpaca.broker.client.BrokerClient` so the rest of momentum_live
    only ever sees the four operations we actually use:

    - :meth:`ensure_subaccount` - get-or-create the sub-account for a pair.
    - :meth:`get_account_cash` - cash balance for a sub-account.
    - :meth:`journal_cash` - JNLC transfer between firm and sub-account.
    - :meth:`reset_subaccount` - cancel orders, close positions, journal back to target.
    """

    DEFAULT_RESET_WAIT_TIMEOUT = 30.0
    DEFAULT_RESET_WAIT_INTERVAL = 1.0
    DEFAULT_JOURNAL_WAIT_TIMEOUT = 20.0
    DEFAULT_JOURNAL_WAIT_INTERVAL = 0.5

    def __init__(
        self,
        credentials: BrokerCredentials,
        registry: BrokerAccountRegistry,
        *,
        broker_client=None,
        http_session=None,
    ):
        self._credentials = credentials
        self._registry = registry
        registry.load()

        if broker_client is None:
            from alpaca.broker.client import BrokerClient

            kwargs: dict = {
                "api_key": credentials.api_key,
                "secret_key": credentials.secret_key,
                "sandbox": credentials.is_sandbox,
            }
            if not credentials.is_sandbox or "sandbox" not in credentials.base_url.lower():
                kwargs["url_override"] = credentials.base_url
            self._client = BrokerClient(**kwargs)
        else:
            self._client = broker_client

        # Raw HTTP session for operations ``alpaca-py`` doesn't expose (e.g.
        # ``PATCH /v1/accounts/{id}`` with ``enabled_assets``). Injected in
        # tests; lazily constructed in production so the requests dependency
        # isn't required when callers skip crypto-enablement.
        self._http_session = http_session

        LOGGER.info(
            "BrokerAccountManager ready (base_url=%s, funding_account=%s, registry=%s)",
            credentials.base_url,
            credentials.funding_account_id or "<unset>",
            registry.path,
        )

    @property
    def client(self):
        """Underlying ``BrokerClient`` (exposed for the per-account trading shim)."""
        return self._client

    @property
    def registry(self) -> BrokerAccountRegistry:
        return self._registry

    @property
    def funding_account_id(self) -> str:
        return self._credentials.funding_account_id

    def ensure_subaccount(self, pair: str, *, label_prefix: str = "bot") -> SubAccountEntry:
        """Return the registry entry for ``pair``, creating the Broker sub-account if needed.

        Also verifies the sub-account has crypto trading enabled (via a raw
        ``PATCH /v1/accounts/{id}`` when necessary). This is idempotent and
        safe for both freshly-created and adopted accounts.
        """
        existing = self._registry.get(pair)
        if existing is not None:
            LOGGER.info("Sub-account for %s already registered: %s", pair, existing.account_id)
            self.ensure_crypto_enabled(existing.account_id)
            return existing

        label = f"{label_prefix}-{pair.replace('/', '')}"
        LOGGER.info("Creating new sandbox sub-account for %s (label=%s)", pair, label)
        account_id = self._create_sandbox_account(label)
        entry = SubAccountEntry.new(pair=pair, account_id=account_id, label=label)
        self._registry.set(entry)
        LOGGER.info("Registered sub-account %s -> %s", pair, account_id)
        self.ensure_crypto_enabled(account_id)
        return entry

    def adopt_subaccount(self, pair: str, account_id: str, *, label: str | None = None) -> SubAccountEntry:
        """Upsert an existing Broker sub-account id into the registry for ``pair``.

        Used by ``--adopt BTC/USD:<uuid>`` to reuse an account that was created
        out-of-band (Brokerdash tutorial, manual bootstrap, etc.) without
        hitting the brittle sandbox account-creation flow.
        """
        try:
            uuid.UUID(str(account_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"adopt_subaccount: {account_id!r} is not a valid UUID") from exc

        resolved_label = label or f"adopted-{pair.replace('/', '')}"
        entry = SubAccountEntry.new(pair=pair, account_id=str(account_id), label=resolved_label)
        self._registry.set(entry)
        LOGGER.info("Adopted sub-account %s -> %s (label=%s)", pair, account_id, resolved_label)
        self.ensure_crypto_enabled(str(account_id))
        return entry

    def list_subaccounts(self) -> dict[str, SubAccountEntry]:
        return self._registry.all()

    def ensure_crypto_enabled(self, account_id: str) -> bool:
        """Make sure ``account_id`` has ``enabled_assets=['us_equity','crypto']``.

        The ``alpaca-py`` ``UpdateAccountRequest`` model does not expose
        ``enabled_assets``, so we issue a raw ``PATCH /v1/accounts/{id}`` with
        HTTP Basic auth (the same credentials the BrokerClient uses).

        Returns ``True`` if a PATCH was issued, ``False`` if the account was
        already crypto-enabled (or we couldn't determine its state safely).

        .. warning::
            **PATCH only updates account metadata.** Alpaca enables crypto
            *trading* in the order-execution engine only when the account is
            **created** with ``enabled_assets`` containing ``crypto``. If you
            adopt a sub-account that was originally created as ``us_equity``
            only, this method will make ``GET /v1/accounts/{id}`` *report*
            crypto-enabled, but ``POST /v1/trading/accounts/{id}/orders`` will
            still reject crypto orders with HTTP 422
            ``crypto orders not allowed for account``. In that case, create a
            fresh sub-account via :meth:`ensure_subaccount` (which sets
            ``enabled_assets`` on ``CreateAccountRequest``) and migrate funds.
            ``test_broker_smoke.py::test_trading_engine_actually_accepts_crypto_orders``
            is the canonical probe for this state.
        """
        session = self._get_http_session()
        base = self._credentials.base_url.rstrip("/")
        auth = (self._credentials.api_key, self._credentials.secret_key)
        url = f"{base}/v1/accounts/{account_id}"

        try:
            resp = session.get(url, auth=auth, timeout=10)
        except Exception as exc:  # pragma: no cover - network
            LOGGER.warning("ensure_crypto_enabled: GET %s failed: %s", url, exc)
            return False
        if resp.status_code != 200:
            LOGGER.warning(
                "ensure_crypto_enabled: GET %s returned %d: %s",
                url,
                resp.status_code,
                resp.text[:200],
            )
            return False

        try:
            payload = resp.json()
        except ValueError:  # pragma: no cover - malformed server response
            LOGGER.warning("ensure_crypto_enabled: non-JSON response for %s", account_id)
            return False

        enabled = [str(x).lower() for x in (payload.get("enabled_assets") or [])]
        crypto_status = str(payload.get("crypto_status") or "").upper()
        if "crypto" in enabled and crypto_status in {"APPROVED", "ACTIVE", ""}:
            LOGGER.debug(
                "ensure_crypto_enabled: %s already crypto-enabled (status=%s)",
                account_id,
                crypto_status or "unset",
            )
            return False

        LOGGER.info(
            "ensure_crypto_enabled: patching %s (enabled_assets=%s, crypto_status=%s)",
            account_id,
            enabled,
            crypto_status or "unset",
        )
        try:
            patch_resp = session.patch(
                url,
                auth=auth,
                json={"enabled_assets": ["us_equity", "crypto"]},
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - network
            LOGGER.warning("ensure_crypto_enabled: PATCH %s failed: %s", url, exc)
            return False
        if patch_resp.status_code >= 400:
            LOGGER.warning(
                "ensure_crypto_enabled: PATCH %s returned %d: %s",
                url,
                patch_resp.status_code,
                patch_resp.text[:200],
            )
            return False
        LOGGER.info("ensure_crypto_enabled: %s patched successfully", account_id)
        return True

    def _get_http_session(self):
        if self._http_session is not None:
            return self._http_session
        import requests

        self._http_session = requests.Session()
        return self._http_session

    def get_account_cash(self, account_id: str) -> float:
        trade_account = self._client.get_trade_account_by_id(account_id)
        return float(trade_account.cash)

    def journal_cash(
        self,
        account_id: str,
        amount: float,
        *,
        direction: str,
        description: str | None = None,
        wait: bool = True,
    ) -> str:
        """Move ``amount`` USD between the firm and ``account_id``.

        Parameters
        ----------
        direction:
            ``"to_sub"`` moves cash from the firm account into ``account_id``;
            ``"from_sub"`` moves cash from ``account_id`` back into the firm
            account. Must be lowercase.
        wait:
            Block until the journal reaches a terminal status. Sandbox journals
            usually clear within seconds.
        """
        from alpaca.broker.enums import JournalEntryType
        from alpaca.broker.requests import CreateJournalRequest

        if amount <= 0:
            raise ValueError("journal amount must be positive")
        direction = direction.lower()
        if direction not in {"to_sub", "from_sub"}:
            raise ValueError(f"direction must be 'to_sub' or 'from_sub', got {direction!r}")
        if not self._credentials.funding_account_id:
            raise RuntimeError(
                "journal_cash requires ALPACA_BROKER_ACCOUNT_ID (the firm/funding "
                "account id). Run with --reset-mode none if you do not have one."
            )

        from_acct, to_acct = (
            (self._credentials.funding_account_id, account_id)
            if direction == "to_sub"
            else (account_id, self._credentials.funding_account_id)
        )

        request = CreateJournalRequest(
            from_account=from_acct,
            to_account=to_acct,
            entry_type=JournalEntryType.CASH,
            amount=Decimal(f"{amount:.2f}"),
            description=description or f"momentum-live {direction} {amount:.2f}",
        )
        LOGGER.info(
            "JNLC %s | from=%s to=%s amount=%.2f",
            direction,
            from_acct,
            to_acct,
            amount,
        )
        try:
            journal = self._client.create_journal(request)
        except Exception as exc:
            LOGGER.error(
                "create_journal failed (amount=%.2f, direction=%s): %s. "
                "If amount > sandbox JNLC Transaction Limit, raise it in the Broker UI.",
                amount,
                direction,
                exc,
            )
            raise

        journal_id = str(getattr(journal, "id", ""))
        if not journal_id:
            raise RuntimeError("create_journal returned no id")

        if wait:
            self._wait_for_journal(journal_id)
        return journal_id

    def reset_subaccount(
        self,
        account_id: str,
        target_balance: float,
        *,
        wait_timeout: float | None = None,
        wait_interval: float | None = None,
        skip_threshold: float = 0.01,
    ) -> dict[str, object]:
        """Soft-reset a sub-account to ``target_balance``.

        Steps: cancel orders -> close positions -> wait for fills -> JNLC the
        cash delta. Returns a small summary dict (initial cash, final cash,
        journal id if any).
        """
        timeout = wait_timeout if wait_timeout is not None else self.DEFAULT_RESET_WAIT_TIMEOUT
        interval = wait_interval if wait_interval is not None else self.DEFAULT_RESET_WAIT_INTERVAL

        LOGGER.info("Resetting sub-account %s to %.2f USD", account_id, target_balance)

        try:
            self._client.cancel_orders_for_account(account_id)
        except Exception as exc:
            LOGGER.warning("cancel_orders_for_account failed for %s: %s", account_id, exc)

        try:
            self._client.close_all_positions_for_account(account_id, cancel_orders=True)
        except Exception as exc:
            LOGGER.warning("close_all_positions_for_account failed for %s: %s", account_id, exc)

        self._wait_for_positions_empty(account_id, timeout=timeout, interval=interval)

        cash_before = self.get_account_cash(account_id)
        delta = round(target_balance - cash_before, 2)
        journal_id: str | None = None

        if abs(delta) < skip_threshold:
            LOGGER.info(
                "Sub-account %s cash %.2f already within %.2f of target; no journal",
                account_id,
                cash_before,
                skip_threshold,
            )
        elif delta > 0:
            journal_id = self.journal_cash(account_id, delta, direction="to_sub", description="reset top-up")
        else:
            journal_id = self.journal_cash(account_id, -delta, direction="from_sub", description="reset skim")

        cash_after = self.get_account_cash(account_id)
        LOGGER.info(
            "Reset complete | account=%s | cash %.2f -> %.2f (target %.2f)",
            account_id,
            cash_before,
            cash_after,
            target_balance,
        )
        return {
            "account_id": account_id,
            "cash_before": cash_before,
            "cash_after": cash_after,
            "target_balance": target_balance,
            "journal_id": journal_id,
        }

    def _wait_for_positions_empty(self, account_id: str, *, timeout: float, interval: float) -> None:
        deadline = time.monotonic() + max(timeout, interval)
        last_count: int | None = None
        while time.monotonic() < deadline:
            try:
                positions = self._client.get_all_positions_for_account(account_id)
            except Exception as exc:
                LOGGER.warning("get_all_positions_for_account failed for %s: %s", account_id, exc)
                return
            count = len(list(positions or []))
            if count == 0:
                return
            if count != last_count:
                LOGGER.info("Waiting on %d open position(s) on %s", count, account_id)
                last_count = count
            time.sleep(interval)
        LOGGER.warning("Timed out after %.1fs waiting for positions to close on %s", timeout, account_id)

    def _wait_for_journal(self, journal_id: str) -> None:
        deadline = time.monotonic() + self.DEFAULT_JOURNAL_WAIT_TIMEOUT
        last_status: str | None = None
        while time.monotonic() < deadline:
            try:
                journal = self._client.get_journal_by_id(journal_id)
            except Exception as exc:
                LOGGER.warning("get_journal_by_id(%s) failed: %s", journal_id, exc)
                return
            status_raw = getattr(journal, "status", None)
            status = str(status_raw.value if hasattr(status_raw, "value") else status_raw or "").lower()
            if status != last_status:
                LOGGER.info("Journal %s status=%s", journal_id, status or "unknown")
                last_status = status
            if status in {"executed", "rejected", "canceled", "refused"}:
                if status != "executed":
                    raise RuntimeError(f"Journal {journal_id} ended in non-executed status: {status}")
                return
            time.sleep(self.DEFAULT_JOURNAL_WAIT_INTERVAL)
        LOGGER.warning("Journal %s did not reach terminal status within %.1fs", journal_id, self.DEFAULT_JOURNAL_WAIT_TIMEOUT)

    def _create_sandbox_account(self, label: str) -> str:
        """Create a sandbox sub-account with deterministic dummy KYC."""
        from alpaca.broker.enums import (
            AgreementType,
            EmploymentStatus,
            FundingSource,
            TaxIdType,
        )
        from alpaca.broker.models import Agreement, Contact, Disclosures, Identity
        from alpaca.broker.requests import CreateAccountRequest

        slug = label.lower().replace("/", "")
        suffix = uuid.uuid4().hex[:8]
        email = f"{slug}-{suffix}@momentum-trader.local"
        # Sandbox rejects SSNs whose digits are all identical after stripping
        # punctuation (e.g. "555-55-5555"). Derive a deterministic-but-varied
        # SSN from the per-account UUID suffix so each KYC payload is unique
        # and passes the API's basic sanity check.
        digits = "".join(c for c in suffix if c.isdigit()).ljust(9, "0")[:9]
        if len(set(digits)) < 2:
            digits = digits[:-1] + ("1" if digits[-1] != "1" else "2")
        ssn = f"{digits[0:3]}-{digits[3:5]}-{digits[5:9]}"

        contact = Contact(
            email_address=email,
            phone_number="+15551234567",
            street_address=["100 Main Street"],
            city="San Francisco",
            state="CA",
            postal_code="94105",
            country="USA",
        )
        identity = Identity(
            given_name=label,
            family_name="Bot",
            date_of_birth=date(1990, 1, 1).isoformat(),
            tax_id=ssn,
            tax_id_type=TaxIdType.USA_SSN,
            country_of_citizenship="USA",
            country_of_birth="USA",
            country_of_tax_residence="USA",
            funding_source=[FundingSource.EMPLOYMENT_INCOME],
        )
        disclosures = Disclosures(
            is_control_person=False,
            is_affiliated_exchange_or_finra=False,
            is_politically_exposed=False,
            immediate_family_exposed=False,
            employment_status=EmploymentStatus.EMPLOYED,
            employer_name="momentum-trader",
            employment_position="bot",
        )
        signed_at = datetime.now(UTC) - timedelta(seconds=5)
        agreements = [
            Agreement(
                agreement=AgreementType.CUSTOMER,
                signed_at=signed_at.isoformat(),
                ip_address="127.0.0.1",
            ),
            Agreement(
                agreement=AgreementType.ACCOUNT,
                signed_at=signed_at.isoformat(),
                ip_address="127.0.0.1",
            ),
            Agreement(
                agreement=AgreementType.MARGIN,
                signed_at=signed_at.isoformat(),
                ip_address="127.0.0.1",
            ),
            Agreement(
                agreement=AgreementType.CRYPTO,
                signed_at=signed_at.isoformat(),
                ip_address="127.0.0.1",
            ),
        ]

        from alpaca.trading.enums import AssetClass

        # Without ``enabled_assets`` the sub-account defaults to ``us_equity`` only
        # (see https://docs.alpaca.markets/reference/createaccount), which makes
        # it unusable for a crypto bot. Explicitly enable crypto trading.
        request = CreateAccountRequest(
            contact=contact,
            identity=identity,
            disclosures=disclosures,
            agreements=agreements,
            enabled_assets=[AssetClass.US_EQUITY, AssetClass.CRYPTO],
        )
        account = self._client.create_account(request)
        account_id = str(getattr(account, "id", ""))
        if not account_id:
            raise RuntimeError("create_account returned no id")
        LOGGER.info("Created sandbox account %s (label=%s)", account_id, label)
        return account_id


__all__ = ["BrokerAccountManager", "BrokerCredentials"]
