import hashlib
import random

from convintel.contracts import Sentiment

from .schema import EntityType, Scenario, Slot

BANKING77_REVISION = "796a4623935746f71378f0ebd435635a8ce08e50"
INTENTS = (
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)
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
    moods = list(range(5)) * 30
    resolutions = [False, True] * 75
    rng.shuffle(moods)
    rng.shuffle(resolutions)
    result: list[Scenario] = []
    for index in range(150):
        sample_seed = int.from_bytes(hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8])
        local = random.Random(sample_seed)
        channel = "voice" if index % 3 == 2 else "chat"
        count = local.randint(3, 5)
        negative, neutral, positive = Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE
        trajectories = (
            (negative,) * (count - 2) + (positive, positive),
            (neutral,) * (count - 2) + (negative, negative),
            (neutral,) * count,
            (negative,) * count,
            (positive,) * count,
        )
        kind_index = (index // 3 + 3) % 6 if channel == "voice" else index % 6
        kinds = (
            () if index % 5 == 0 else (ENTITY_TYPES[kind_index], ENTITY_TYPES[(kind_index + 1) % 6])
        )
        result.append(
            Scenario(
                id=f"synth-v1-{seed}-{index:03d}",
                index=index,
                seed=seed,
                sample_seed=sample_seed,
                intent=next(voice if channel == "voice" else chat),
                channel=channel,
                customer_turns=count,
                customer_trajectory=trajectories[moods[index]],
                resolved=resolutions[index],
                escalation_requested=index % 5 == 1,
                persona=local.choice(
                    ("new to mobile banking", "experienced user", "busy customer")
                ),
                verbosity=local.choice(("brief", "moderate")),
                noise=local.choice(("none", "light")),
                slots=tuple(Slot(entity_type=kind, token=f"[[{kind}_1]]") for kind in kinds),
            )
        )
    return tuple(result)
