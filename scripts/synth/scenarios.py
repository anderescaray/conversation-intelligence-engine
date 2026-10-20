import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import cast

from convintel.contracts import Sentiment
from convintel.errors import ConvIntelError

from .plausibility import (
    ESCALATION_KINDS,
    INTENT_KIND,
    LAST_FOUR_PII,
    PII_FAMILY,
    RULES,
    IntentKind,
    PlanVersion,
    allowed_pii,
)
from .schema import EntityType, Placement, Record, Scenario, ScenarioV3

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


def legacy_scenarios(seed: int) -> tuple[Scenario, ...]:
    """Scenario plan v2, used by generators 2.0 and 3.0; reproduces their stored candidates."""
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


PERSONAS = ("new to mobile banking", "experienced user", "busy customer")
MAX_REROLLS = 2


def sample_seed(seed: int, index: int, reroll: int = 0) -> int:
    key = f"{seed}:{index}" if reroll == 0 else f"{seed}:{index}:reroll{reroll}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8])


def free_attributes(seed: int) -> dict[str, object]:
    """Attributes a same-stratum reroll may change; labels and PII plan are never among them."""
    local = random.Random(seed)
    return {
        "customer_turns": local.randint(3, 4),
        "persona": local.choice(PERSONAS),
        "verbosity": local.choice(("brief", "moderate")),
        "noise": local.choice(("none", "light")),
    }


def _weighted(rng: random.Random, weights: dict[Sentiment, int]) -> Sentiment:
    options = list(weights)
    return rng.choices(options, [weights[option] for option in options])[0]


def _assign_pii(intents: list[str], rng: random.Random) -> dict[int, list[EntityType]]:
    """Two plausible full PII types per non-control scenario, balancing counts across types.

    Scenarios with the fewest plausible types choose first, so constrained intents are not left
    with only over-represented types; ties are broken by the seeded generator.
    """
    indices = [index for index in range(len(intents)) if index % 5 != 0]
    keys = {index: rng.random() for index in indices}
    counts: Counter[str] = Counter()
    chosen: dict[int, list[EntityType]] = {}
    for index in sorted(indices, key=lambda i: (len(allowed_pii(intents[i])), keys[i])):
        options = list(allowed_pii(intents[index]))
        rng.shuffle(options)
        options.sort(key=lambda name: counts[name])
        pair: list[EntityType] = [cast(EntityType, name) for name in options[:2]]
        rng.shuffle(pair)
        counts.update(pair)
        chosen[index] = pair
    return chosen


def scenarios(seed: int, plan: PlanVersion = 4) -> tuple[ScenarioV3, ...]:
    """Plans v3/v4 for golden-v1: v2 channel and intent quotas with plausible labels and PII.

    Plan 4 is current (generator 3.2); plan 3 reproduces the generator 3.1 pilot.
    """
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = random.Random(seed)
    intents = list(INTENTS)
    rng.shuffle(intents)
    chat = iter([*intents, *intents[:23]])
    voice = iter(intents[23:73])
    channels = ["voice" if index % 3 == 2 else "chat" for index in range(150)]
    assigned = [next(voice if channel == "voice" else chat) for channel in channels]
    kinds: list[IntentKind] = [INTENT_KIND[intent] for intent in assigned]
    rules = RULES[plan]
    if plan == 3:
        resolutions = [False, True] * 75
        rng.shuffle(resolutions)
    else:
        # Kinds that are always answered take part of the resolved quota; the 75 unresolved
        # targets fall on the remaining intents, keeping 75/75 overall.
        fixed = [kinds[i] in rules.always_resolved_kinds for i in range(150)]
        free = [True] * (75 - sum(fixed)) + [False] * 75
        rng.shuffle(free)
        remaining = iter(free)
        resolutions = [True if fixed[i] else next(remaining) for i in range(150)]
    escalated: set[int] = set()
    for channel, count in (("chat", 20), ("voice", 10)):
        eligible = [
            i for i in range(150) if channels[i] == channel and kinds[i] in ESCALATION_KINDS
        ]
        escalated.update(rng.sample(eligible, count))
    rare_positive = set(
        rng.sample(
            [
                i
                for i in range(150)
                if not resolutions[i]
                and i not in escalated
                and kinds[i] in rules.rare_positive_kinds
            ],
            3,
        )
    )
    pii = _assign_pii(assigned, rng)
    confirmable = sorted(i for i, kinds_ in pii.items() if LAST_FOUR_PII & set(kinds_))
    partial = {
        index: rng.choice([kind for kind in pii[index] if kind in LAST_FOUR_PII])
        for index in sorted(rng.sample(confirmable, 30))
    }
    result: list[ScenarioV3] = []
    for index in range(150):
        intent, resolved = assigned[index], resolutions[index]
        kind: IntentKind = kinds[index]
        labels = random.Random(f"{seed}:{index}:labels")
        if index in escalated:
            start = Sentiment.NEGATIVE
            end = _weighted(labels, rules.end[kind, True]) if resolved else Sentiment.NEGATIVE
        elif index in rare_positive:
            start, end = Sentiment.NEUTRAL, Sentiment.POSITIVE
        else:
            start = _weighted(labels, rules.start[kind])
            end = _weighted(labels, rules.end[kind, resolved])
        slots = [
            Placement(
                entity_type=entity,
                token=f"[[{entity}_1]]",
                turn_index=2 * position,
                speaker="customer",
            )
            for position, entity in enumerate(pii.get(index, []))
        ]
        if index in partial:
            source = next(slot for slot in slots if slot.entity_type == partial[index])
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
        seed_value = sample_seed(seed, index)
        result.append(
            ScenarioV3.model_validate(
                {
                    "id": f"synth-v{plan}-{seed}-{index:03d}",
                    "plan_version": plan,
                    "index": index,
                    "seed": seed,
                    "sample_seed": seed_value,
                    "intent": intent,
                    "intent_description": INTENT_DESCRIPTIONS[intent],
                    "intent_kind": kind,
                    "pii_family": PII_FAMILY[intent],
                    "channel": channels[index],
                    "start_sentiment": start,
                    "end_sentiment": end,
                    "resolved": resolved,
                    "escalation_requested": index in escalated,
                    "slots": tuple(sorted(slots, key=lambda slot: slot.turn_index)),
                    **free_attributes(seed_value),
                }
            )
        )
    return tuple(result)


def reroll(scenario: ScenarioV3, attempt: int) -> ScenarioV3:
    """Same-stratum replacement: identical labels, intent, channel and PII plan."""
    if not 0 <= attempt <= MAX_REROLLS:
        raise ValueError(f"reroll must be within [0, {MAX_REROLLS}]")
    seed_value = sample_seed(scenario.seed, scenario.index, attempt)
    return ScenarioV3.model_validate(
        {
            **scenario.model_dump(),
            **free_attributes(seed_value),
            "sample_seed": seed_value,
            "reroll": attempt,
        }
    )


class PlanReport(Record):
    plan: str
    seed: int
    size: int
    by_channel: dict[str, int]
    by_intent_kind: dict[str, int]
    by_pii_family: dict[str, int]
    intents_covered: int
    no_pii_controls: int
    full_pii_by_type: dict[str, int]
    full_pii_by_type_and_channel: dict[str, dict[str, int]]
    last_four_by_type: dict[str, int]
    by_resolved: dict[str, int]
    escalation_by_kind: dict[str, int]
    escalation_by_channel: dict[str, int]
    start_by_kind: dict[str, dict[str, int]]
    end_by_kind_and_resolved: dict[str, dict[str, int]]
    sentiment_pairs: dict[str, int]
    rare_positive_unresolved: int
    customer_turns: dict[str, int]


def plan_report(plan: tuple[ScenarioV3, ...]) -> PlanReport:
    def grouped(key: str, values: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
        table: dict[str, Counter[str]] = {}
        for group, value in values:
            table.setdefault(group, Counter())[value] += 1
        return {group: dict(sorted(counter.items())) for group, counter in sorted(table.items())}

    full = [(s, slot) for s in plan for slot in s.slots if slot.exposure == "full"]
    return PlanReport(
        plan=plan[0].plan_prefix(),
        seed=plan[0].seed,
        size=len(plan),
        by_channel=dict(Counter(s.channel for s in plan)),
        by_intent_kind=dict(Counter(s.intent_kind for s in plan)),
        by_pii_family=dict(Counter(s.pii_family for s in plan)),
        intents_covered=len({s.intent for s in plan}),
        no_pii_controls=sum(not s.slots for s in plan),
        full_pii_by_type=dict(Counter(slot.entity_type for _, slot in full)),
        full_pii_by_type_and_channel=grouped(
            "type", [(slot.entity_type, s.channel) for s, slot in full]
        ),
        last_four_by_type=dict(
            Counter(slot.entity_type for s in plan for slot in s.slots if slot.exposure != "full")
        ),
        by_resolved=dict(Counter(str(s.resolved).lower() for s in plan)),
        escalation_by_kind=dict(Counter(s.intent_kind for s in plan if s.escalation_requested)),
        escalation_by_channel=dict(Counter(s.channel for s in plan if s.escalation_requested)),
        start_by_kind=grouped("kind", [(s.intent_kind, s.start_sentiment.value) for s in plan]),
        end_by_kind_and_resolved=grouped(
            "kind",
            [(f"{s.intent_kind}:resolved={s.resolved}", s.end_sentiment.value) for s in plan],
        ),
        sentiment_pairs=dict(
            Counter(f"{s.start_sentiment.value}->{s.end_sentiment.value}" for s in plan)
        ),
        rare_positive_unresolved=sum(
            not s.resolved and not s.escalation_requested and s.end_sentiment == "positive"
            for s in plan
        ),
        customer_turns=dict(Counter(str(s.customer_turns) for s in plan)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the current scenario plan report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(1, "Output exists; plan reports are immutable\n")
    try:
        report = plan_report(scenarios(args.seed))
    except (ValueError, ConvIntelError) as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
