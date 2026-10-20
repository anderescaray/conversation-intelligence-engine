"""Code-owned plausibility rules for scenario plan v3.

Each Banking77 intent gets a kind (what the customer wants) and a PII family (which personal
data a customer would plausibly volunteer for it). Labels and PII types are only sampled inside
these rules, so the generator is never asked for a contradiction such as an IBAN in a card
top-up or an angry opening to a product question. The tables cover the 77 intents explicitly so
that every assignment is reviewable; a test enforces that coverage.
"""

from typing import Literal, NamedTuple

from convintel.contracts import Sentiment

IntentKind = Literal["info", "problem", "request"]
PIIFamily = Literal["card", "bank", "topup", "account", "general"]

NEGATIVE, NEUTRAL, POSITIVE = Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE

_KINDS: dict[IntentKind, str] = {
    "info": """
        age_limit atm_support card_acceptance card_delivery_estimate country_support
        disposable_card_limits exchange_charge exchange_rate fiat_currency_support
        order_physical_card receiving_money supported_cards_and_currencies
        top_up_by_bank_transfer_charge top_up_by_card_charge top_up_by_cash_or_cheque
        top_up_limits topping_up_by_card transfer_into_account transfer_timing
        verify_my_identity visa_or_mastercard why_verify_identity
    """,
    "request": """
        activate_my_card apple_pay_or_google_pay automatic_top_up cancel_transfer
        card_about_to_expire card_linking change_pin edit_personal_details exchange_via_app
        get_disposable_virtual_card get_physical_card getting_spare_card getting_virtual_card
        passcode_forgotten request_refund terminate_account verify_source_of_funds verify_top_up
    """,
    "problem": """
        Refund_not_showing_up balance_not_updated_after_bank_transfer
        balance_not_updated_after_cheque_or_cash_deposit beneficiary_not_allowed card_arrival
        card_not_working card_payment_fee_charged card_payment_not_recognised
        card_payment_wrong_exchange_rate card_swallowed cash_withdrawal_charge
        cash_withdrawal_not_recognised compromised_card contactless_not_working
        declined_card_payment declined_cash_withdrawal declined_transfer
        direct_debit_payment_not_recognised extra_charge_on_statement failed_transfer
        lost_or_stolen_card lost_or_stolen_phone pending_card_payment pending_cash_withdrawal
        pending_top_up pending_transfer pin_blocked reverted_card_payment? top_up_failed
        top_up_reverted transaction_charged_twice transfer_fee_charged
        transfer_not_received_by_recipient unable_to_verify_identity virtual_card_not_working
        wrong_amount_of_cash_received wrong_exchange_rate_for_cash_withdrawal
    """,
}
_FAMILIES: dict[PIIFamily, str] = {
    "card": """
        Refund_not_showing_up activate_my_card apple_pay_or_google_pay automatic_top_up
        card_about_to_expire card_arrival card_linking card_not_working card_payment_fee_charged
        card_payment_not_recognised card_payment_wrong_exchange_rate card_swallowed
        cash_withdrawal_charge cash_withdrawal_not_recognised change_pin compromised_card
        contactless_not_working declined_card_payment declined_cash_withdrawal
        extra_charge_on_statement getting_spare_card lost_or_stolen_card pending_card_payment
        pending_cash_withdrawal pin_blocked request_refund reverted_card_payment?
        top_up_by_card_charge transaction_charged_twice verify_top_up virtual_card_not_working
        wrong_amount_of_cash_received wrong_exchange_rate_for_cash_withdrawal
    """,
    "bank": """
        balance_not_updated_after_bank_transfer beneficiary_not_allowed cancel_transfer
        declined_transfer direct_debit_payment_not_recognised failed_transfer pending_transfer
        receiving_money top_up_by_bank_transfer_charge transfer_fee_charged
        transfer_into_account transfer_not_received_by_recipient
    """,
    "topup": "pending_top_up top_up_failed top_up_reverted",
    "account": """
        balance_not_updated_after_cheque_or_cash_deposit edit_personal_details
        lost_or_stolen_phone passcode_forgotten terminate_account top_up_by_cash_or_cheque
        top_up_limits unable_to_verify_identity verify_my_identity verify_source_of_funds
        why_verify_identity
    """,
    "general": """
        age_limit atm_support card_acceptance card_delivery_estimate country_support
        disposable_card_limits exchange_charge exchange_rate exchange_via_app
        fiat_currency_support get_disposable_virtual_card get_physical_card getting_virtual_card
        order_physical_card supported_cards_and_currencies topping_up_by_card transfer_timing
        visa_or_mastercard
    """,
}
INTENT_KIND: dict[str, IntentKind] = {
    intent: kind for kind, names in _KINDS.items() for intent in names.split()
}
PII_FAMILY: dict[str, PIIFamily] = {
    intent: family for family, names in _FAMILIES.items() for intent in names.split()
}

# Identity checks are plausible for any intent; instrument data only where the intent uses it.
IDENTITY_PII = ("PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS")
FAMILY_PII: dict[PIIFamily, tuple[str, ...]] = {
    "card": ("CREDIT_CARD",),
    "bank": ("IBAN_CODE", "ACCOUNT_NUMBER"),
    "topup": ("CREDIT_CARD", "IBAN_CODE"),
    "account": ("ACCOUNT_NUMBER",),
    "general": (),
}
# Closing an account returns the balance to, and source-of-funds checks name, an external account.
INTENT_EXTRA_PII: dict[str, tuple[str, ...]] = {
    "terminate_account": ("IBAN_CODE",),
    "verify_source_of_funds": ("IBAN_CODE",),
}
LAST_FOUR_PII = frozenset({"CREDIT_CARD", "IBAN_CODE", "PHONE_NUMBER", "ACCOUNT_NUMBER"})


def allowed_pii(intent: str) -> tuple[str, ...]:
    return (
        *IDENTITY_PII,
        *FAMILY_PII[PII_FAMILY[intent]],
        *INTENT_EXTRA_PII.get(intent, ()),
    )


PlanVersion = Literal[3, 4]


class LabelRules(NamedTuple):
    """Sampling weights for one scenario plan; a target is plausible iff its weight exists."""

    start: dict[IntentKind, dict[Sentiment, int]]
    end: dict[tuple[IntentKind, bool], dict[Sentiment, int]]
    rare_positive_kinds: frozenset[IntentKind]
    always_resolved_kinds: frozenset[IntentKind]


ESCALATION_KINDS: frozenset[IntentKind] = frozenset({"problem", "request"})
_PROBLEM_REQUEST_START: dict[IntentKind, dict[Sentiment, int]] = {
    "problem": {NEGATIVE: 3, NEUTRAL: 1},
    "request": {NEUTRAL: 6, NEGATIVE: 3, POSITIVE: 1},
}
_PROBLEM_REQUEST_END: dict[tuple[IntentKind, bool], dict[Sentiment, int]] = {
    ("problem", True): {POSITIVE: 4, NEUTRAL: 1},
    ("problem", False): {NEGATIVE: 7, NEUTRAL: 3},
    ("request", True): {POSITIVE: 3, NEUTRAL: 1},
    ("request", False): {NEGATIVE: 1, NEUTRAL: 1},
}
# Plan 3 (generator 3.1): a product question never opens angrily and never escalates; a reported
# problem never opens happily. Its pilot showed informational questions still failing every
# attempt on an unresolved question, a positive opening or a neutral close.
# Plan 4 (generator 3.2): informational questions open neutral, are always answered and close
# positive, so golden-v1 contains no unresolved informational question.
RULES: dict[PlanVersion, LabelRules] = {
    3: LabelRules(
        start={"info": {NEUTRAL: 4, POSITIVE: 1}, **_PROBLEM_REQUEST_START},
        end={
            ("info", True): {POSITIVE: 1, NEUTRAL: 1},
            ("info", False): {NEUTRAL: 3, NEGATIVE: 2},
            **_PROBLEM_REQUEST_END,
        },
        rare_positive_kinds=frozenset({"info", "request"}),
        always_resolved_kinds=frozenset(),
    ),
    4: LabelRules(
        start={"info": {NEUTRAL: 1}, **_PROBLEM_REQUEST_START},
        end={("info", True): {POSITIVE: 1}, **_PROBLEM_REQUEST_END},
        rare_positive_kinds=frozenset({"request"}),
        always_resolved_kinds=frozenset({"info"}),
    ),
}


def label_violations(
    intent: str,
    *,
    resolved: bool,
    escalation_requested: bool,
    start: Sentiment,
    end: Sentiment,
    plan: PlanVersion = 4,
) -> list[str]:
    """Return why a label combination is implausible for this intent; empty when allowed."""
    rules = RULES[plan]
    kind = INTENT_KIND[intent]
    errors: list[str] = []
    if escalation_requested and kind not in ESCALATION_KINDS:
        errors.append(f"{kind} intents never escalate")
    if escalation_requested and start != NEGATIVE:
        errors.append("escalation must start negative")
    if escalation_requested and not resolved and end != NEGATIVE:
        errors.append("unresolved escalation must end negative")
    if not resolved and kind in rules.always_resolved_kinds:
        errors.append(f"{kind} intents are always resolved")
    if start not in rules.start[kind] and not escalation_requested:
        errors.append(f"{kind} intents cannot start {start.value}")
    rare = (
        not resolved
        and not escalation_requested
        and kind in rules.rare_positive_kinds
        and (start, end) == (NEUTRAL, POSITIVE)
    )
    unresolved_escalation = escalation_requested and not resolved
    allowed_end = rules.end.get((kind, resolved), {})
    if not unresolved_escalation and end not in allowed_end and not rare:
        errors.append(f"{kind} intents cannot end {end.value} when resolved={resolved}")
    return errors
