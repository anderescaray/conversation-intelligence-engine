"""Operational descriptions of the 77 Banking77 intents, versioned and immutable.

One source of truth for every consumer that shows the taxonomy to a model or a person: the
synthetic-data generator and judge (T06), the golden-set labeling guide and the LLM intent
verifier (T10). The golden set was labeled against these definitions and will evaluate the
verifier, so both must read the same strings.

Descriptions are versioned and immutable for the same reason prompts are (ADR-0004): editing
one silently changes what earlier data was generated, judged or verified against. Changing a
description means adding a version, never editing an existing one, and every caller pins a
version explicitly; there is no default.

- Version 1: the first one-line definition of each intent.
- Version 2: audit against Banking77 train examples; rewrites descriptions that contradicted
  their examples, disambiguates confusable families and rewords some so that none shares six
  consecutive words with a Banking77 utterance.
- Version 3: version 2 with exactly two intents changed, ``extra_charge_on_statement`` and
  ``card_payment_not_recognised``, separated by what their train examples actually say. It is
  the version the LLM verifier uses.

Descriptions paraphrase; they never quote an utterance. Labels keep Banking77's exact names,
including ``Refund_not_showing_up`` and ``reverted_card_payment?``.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, get_args

DescriptionsVersion = Literal[1, 2, 3]
DESCRIPTIONS_VERSIONS: tuple[DescriptionsVersion, ...] = get_args(DescriptionsVersion)

# Provenance, not a dataset pin: the Banking77 repo and revision whose label names and train
# examples these descriptions were audited against (as pinned by the T06 taxonomy). The
# product's Banking77 is the classifier's, pinned in training/intent/data.py
# (mteb/banking77@18072d2685ea...); never load data from these constants. A revision without its
# repo is ambiguous, and the two pins live in different repos.
AUDITED_AGAINST_REPO = "PolyAI/banking77"
AUDITED_AGAINST_REVISION = "796a4623935746f71378f0ebd435635a8ce08e50"

_V1: dict[str, str] = {
    "activate_my_card": "Activating a newly received bank card so it can be used.",
    "age_limit": "The minimum age required to open or use a banking account.",
    "apple_pay_or_google_pay": "Setting up or using a bank card with Apple Pay or Google Pay.",
    "atm_support": "Which ATMs or ATM networks support cash withdrawals with the card.",
    "automatic_top_up": "Configuring or troubleshooting automatic account top-ups.",
    "balance_not_updated_after_bank_transfer": (
        "An incoming bank transfer has completed but the account balance has not updated."
    ),
    "balance_not_updated_after_cheque_or_cash_deposit": (
        "A cash or cheque deposit has not appeared in the account balance."
    ),
    "beneficiary_not_allowed": (
        "A bank transfer recipient is blocked or not permitted as a beneficiary."
    ),
    "cancel_transfer": "Cancelling an already initiated bank transfer.",
    "card_about_to_expire": "Replacing or renewing a card that is approaching its expiry date.",
    "card_acceptance": "Whether a physical card is accepted by particular merchants or locations.",
    "card_arrival": "Locating an ordered physical card that has not arrived.",
    "card_delivery_estimate": "Learning the expected delivery time for a physical card.",
    "card_linking": "Linking an existing card to the correct account or app.",
    "card_not_working": (
        "A physical card is not working, without a specifically identified payment decline."
    ),
    "card_payment_fee_charged": "A fee was charged for making a card payment.",
    "card_payment_not_recognised": (
        "A card purchase on the statement is not recognised by the customer."
    ),
    "card_payment_wrong_exchange_rate": (
        "A card purchase used an unexpected currency exchange rate."
    ),
    "card_swallowed": "An ATM retained the customer's physical card.",
    "cash_withdrawal_charge": "A fee was charged for a cash withdrawal.",
    "cash_withdrawal_not_recognised": (
        "A cash withdrawal on the statement was not made or recognised by the customer."
    ),
    "change_pin": "Changing the PIN of a bank card.",
    "compromised_card": (
        "Suspected exposure or theft of card details, even if the physical card is still held."
    ),
    "contactless_not_working": "The contactless tap-to-pay function of a card is not working.",
    "country_support": (
        "Whether residents of a particular country can open or use the banking service."
    ),
    "declined_card_payment": "A card purchase was explicitly declined.",
    "declined_cash_withdrawal": "An ATM cash withdrawal was explicitly declined.",
    "declined_transfer": "A bank transfer was explicitly declined or rejected.",
    "direct_debit_payment_not_recognised": (
        "A direct debit payment is not recognised by the customer."
    ),
    "disposable_card_limits": "Usage restrictions or limits of disposable virtual cards.",
    "edit_personal_details": "Changing personal information recorded on the account.",
    "exchange_charge": "Fees charged for exchanging one currency into another.",
    "exchange_rate": "The currency exchange rate offered or how that rate is determined.",
    "exchange_via_app": (
        "Exchanging currencies within the app, not sending money to another person."
    ),
    "extra_charge_on_statement": (
        "An additional unexplained charge appears on the statement, not a duplicate transaction."
    ),
    "failed_transfer": "A bank transfer failed to execute, rather than remaining pending.",
    "fiat_currency_support": "Which traditional currencies the account can hold or use.",
    "get_disposable_virtual_card": "Obtaining a disposable single-use virtual card.",
    "get_physical_card": "Obtaining a physical plastic card rather than relying on a virtual card.",
    "getting_spare_card": (
        "Obtaining an additional spare physical card while keeping the existing one."
    ),
    "getting_virtual_card": "Creating a reusable virtual card in the app.",
    "lost_or_stolen_card": (
        "Reporting a physical card that has been lost or stolen and securing it."
    ),
    "lost_or_stolen_phone": (
        "Securing access to the banking account after a phone was lost or stolen."
    ),
    "order_physical_card": "The steps for placing a physical card order in the app.",
    "passcode_forgotten": (
        "Recovering access after forgetting the app login passcode, not the card PIN."
    ),
    "pending_card_payment": "A card purchase remains in pending status.",
    "pending_cash_withdrawal": "An ATM withdrawal remains in pending status on the account.",
    "pending_top_up": "Adding money to the account is still marked as a pending top-up.",
    "pending_transfer": "A bank transfer remains in pending status.",
    "pin_blocked": "A card PIN has been blocked, usually after unsuccessful PIN attempts.",
    "receiving_money": "How another person can send money to the customer's account.",
    "Refund_not_showing_up": (
        "A merchant has issued a refund but it has not appeared in the account."
    ),
    "request_refund": "Requesting a refund for a transaction, before the refund has been issued.",
    "reverted_card_payment?": (
        "A card payment was reversed or reverted and the customer wants to understand it."
    ),
    "supported_cards_and_currencies": (
        "Which funding cards and currencies are supported when topping up by card."
    ),
    "terminate_account": "Closing the customer's banking account permanently.",
    "top_up_by_bank_transfer_charge": (
        "Charges for funding the account through an incoming bank transfer."
    ),
    "top_up_by_card_charge": "Charges for funding the account with another payment card.",
    "top_up_by_cash_or_cheque": "Whether and how cash or cheques can be used to fund the account.",
    "top_up_failed": "An attempted top-up failed, rather than remaining pending.",
    "top_up_limits": "Minimum or maximum amounts allowed for account top-ups.",
    "top_up_reverted": "A completed or attempted top-up was reversed back to its funding source.",
    "topping_up_by_card": "How to add money to the account using a funding card.",
    "transaction_charged_twice": "The same transaction was charged twice.",
    "transfer_fee_charged": "A fee was charged for an outgoing bank transfer.",
    "transfer_into_account": (
        "Funding the account from an external bank account using bank transfer details."
    ),
    "transfer_not_received_by_recipient": (
        "An outgoing transfer was sent but the intended recipient has not received it."
    ),
    "transfer_timing": "How long a bank transfer normally takes to arrive.",
    "unable_to_verify_identity": (
        "An attempted identity verification cannot be completed or is failing."
    ),
    "verify_my_identity": "How to complete the bank's identity verification process.",
    "verify_source_of_funds": (
        "Providing documents or information to verify where deposited funds came from."
    ),
    "verify_top_up": "Completing a verification or security check for a card top-up.",
    "virtual_card_not_working": "A reusable virtual card is not working.",
    "visa_or_mastercard": "Whether the issued card belongs to the Visa or Mastercard network.",
    "why_verify_identity": "Why identity verification is required by the bank.",
    "wrong_amount_of_cash_received": (
        "An ATM dispensed a different amount of cash than requested or debited."
    ),
    "wrong_exchange_rate_for_cash_withdrawal": (
        "A cash withdrawal used an unexpected currency exchange rate."
    ),
}
_V2_CHANGES: dict[str, str] = {
    "apple_pay_or_google_pay": (
        "Adding money to the account through Apple Pay or Google Pay, or using the card with "
        "those wallets."
    ),
    "get_physical_card": (
        "Finding, receiving or setting up the PIN that belongs to a newly issued physical card."
    ),
    "virtual_card_not_working": (
        "A virtual card, whether reusable or disposable, is not working for payments."
    ),
    "visa_or_mastercard": (
        "Which card network, Visa or Mastercard, the issued card uses, or asking for one "
        "network instead of the other."
    ),
    "card_payment_fee_charged": (
        "An unexpected extra fee applied to a purchase paid with the card, not to a top-up or a "
        "cash withdrawal."
    ),
    "change_pin": ("Changing a working card PIN to a new one, in the app or at a cash machine."),
    "getting_spare_card": (
        "Getting an additional card for the same account while keeping the current one, for "
        "oneself or a family member."
    ),
    "lost_or_stolen_card": (
        "Reporting a physical card as lost or stolen so that it is blocked and replaced."
    ),
    "lost_or_stolen_phone": (
        "Securing the account and regaining app access after the phone used for banking was "
        "lost or stolen."
    ),
    "order_physical_card": (
        "Ordering a physical card to be sent out, including how to request it and what it costs."
    ),
    "passcode_forgotten": (
        "Resetting a forgotten app login passcode while the customer still has their phone."
    ),
    "pin_blocked": ("The card PIN no longer works or was locked after too many wrong attempts."),
    "Refund_not_showing_up": (
        "A merchant says a refund was issued, but the money has not appeared in the account."
    ),
    "request_refund": (
        "Asking to get money back for a purchase, or to cancel it, before any refund has been "
        "issued."
    ),
    "reverted_card_payment?": (
        "A card payment was reversed, cancelled or returned to the account, and the customer "
        "wants to know why."
    ),
    "top_up_by_bank_transfer_charge": (
        "Whether a fee applies when money is added to the account by an incoming bank transfer, "
        "including SEPA or SWIFT."
    ),
    "top_up_by_card_charge": (
        "Whether, and how much, the customer pays to fund the account from a payment card, "
        "often one issued abroad."
    ),
    "topping_up_by_card": (
        "How to fund the account from a payment card, or why a card top-up did not arrive; fee "
        "questions belong elsewhere."
    ),
    "age_limit": ("How old someone must be to open or use an account."),
    "cash_withdrawal_charge": ("A fee appeared after taking out cash at an ATM."),
    "wrong_amount_of_cash_received": (
        "An ATM paid out less or more cash than was requested or debited."
    ),
}
_V3_CHANGES: dict[str, str] = {
    "extra_charge_on_statement": (
        "A small extra charge or fee, often about one unit of currency and sometimes still "
        "pending, shows on the statement and the customer asks what it is for; not a payment to "
        "an unknown merchant, a fee caused by using the card to pay, or a duplicate."
    ),
    "card_payment_not_recognised": (
        "The customer says they did not make a card payment to a merchant, or do not recognise "
        "the merchant, and may fear that someone else is using the card; a small extra charge "
        "of unclear origin belongs elsewhere."
    ),
}

_DESCRIPTIONS: Mapping[DescriptionsVersion, Mapping[str, str]] = MappingProxyType(
    {
        1: MappingProxyType(_V1),
        2: MappingProxyType({**_V1, **_V2_CHANGES}),
        3: MappingProxyType({**_V1, **_V2_CHANGES, **_V3_CHANGES}),
    }
)


def intent_descriptions(version: DescriptionsVersion) -> Mapping[str, str]:
    """Label -> description for one pinned version, read-only, in Banking77 label order."""
    try:
        return _DESCRIPTIONS[version]
    except KeyError:
        raise ValueError(
            f"unknown intent descriptions version {version!r}; known: {DESCRIPTIONS_VERSIONS}"
        ) from None
