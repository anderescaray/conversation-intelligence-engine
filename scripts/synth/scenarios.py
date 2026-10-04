import hashlib
import random

from convintel.contracts import Sentiment

from .schema import EntityType, Placement, Scenario

BANKING77_REVISION = "796a4623935746f71378f0ebd435635a8ce08e50"
INTENT_DESCRIPTIONS = {
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
INTENTS = tuple(INTENT_DESCRIPTIONS)
ENTITY_TYPES: tuple[EntityType, ...] = (
    "PERSON",
    "CREDIT_CARD",
    "IBAN_CODE",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ACCOUNT_NUMBER",
)


def scenarios(seed: int) -> tuple[Scenario, ...]:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = random.Random(seed)
    intents = list(INTENTS)
    rng.shuffle(intents)
    chat = iter([*intents, *intents[:23]])
    voice = iter(intents[23:73])
    resolutions = [False, True] * 75
    rng.shuffle(resolutions)
    rare_positive = set(rng.sample([i for i in range(150) if not resolutions[i] and i % 5 != 1], 3))
    result: list[Scenario] = []
    for index in range(150):
        sample_seed = int.from_bytes(hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8])
        local = random.Random(sample_seed)
        channel = "voice" if index % 3 == 2 else "chat"
        count = local.randint(3, 4)
        negative, neutral, positive = Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE
        escalation = index % 5 == 1
        start = (
            negative if escalation else local.choices((negative, neutral, positive), (5, 4, 1))[0]
        )
        if index in rare_positive:
            start, end = neutral, positive
        elif resolutions[index]:
            end = local.choices((positive, neutral), (4, 1))[0]
        else:
            end = negative if escalation else local.choices((negative, neutral), (7, 3))[0]
        kind_index = (index // 3 + 3) % 6 if channel == "voice" else index % 6
        kinds = (
            () if index % 5 == 0 else (ENTITY_TYPES[kind_index], ENTITY_TYPES[(kind_index + 1) % 6])
        )
        slots = [
            Placement(
                entity_type=kind,
                token=f"[[{kind}_1]]",
                turn_index=2 * position,
                speaker="customer",
            )
            for position, kind in enumerate(kinds)
        ]
        eligible = [
            p
            for p in slots
            if p.entity_type in {"CREDIT_CARD", "IBAN_CODE", "PHONE_NUMBER", "ACCOUNT_NUMBER"}
        ]
        if eligible and index % 4 == 1:
            source = local.choice(eligible)
            slots.append(
                Placement(
                    entity_type=source.entity_type,
                    token=f"[[{source.entity_type}_LAST_FOUR_1]]",
                    turn_index=source.turn_index + 1,
                    speaker="agent",
                    exposure="last_four",
                    source_token=source.token,
                )
            )
        intent = next(voice if channel == "voice" else chat)
        result.append(
            Scenario(
                id=f"synth-v2-{seed}-{index:03d}",
                index=index,
                seed=seed,
                sample_seed=sample_seed,
                intent=intent,
                intent_description=INTENT_DESCRIPTIONS[intent],
                channel=channel,
                customer_turns=count,
                start_sentiment=start,
                end_sentiment=end,
                resolved=resolutions[index],
                escalation_requested=escalation,
                persona=local.choice(
                    ("new to mobile banking", "experienced user", "busy customer")
                ),
                verbosity=local.choice(("brief", "moderate")),
                noise=local.choice(("none", "light")),
                slots=tuple(sorted(slots, key=lambda slot: slot.turn_index)),
            )
        )
    return tuple(result)
