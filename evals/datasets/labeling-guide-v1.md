# Labeling guide v1

## Resolved
- resolved=true: the requested action is completed, or committed with a known outcome where only processing or delivery time remains (a refund processed and on its way; a card ordered; a PIN reset). A correctly answered informational question is resolved.
- resolved=false: the outcome still depends on something undecided or on another contact (an investigation, a dispute decision, a callback, a handover to another team or person); or the agent cannot do it; or the customer leaves without a clear path forward.
- An escalation is resolved only if the person it is escalated to resolves it within the same conversation (for example, a supervisor joins the chat and processes the refund). Passing the case on for a later contact is unresolved.
- Examples:
  - "I've processed your refund; it reaches your account in 5-7 working days" -> resolved.
  - "I've sent your refund request to the team for review" -> unresolved.
  - "I've opened a dispute; we'll decide within 10 working days" -> unresolved.
  - "A supervisor will call you within 2 days" -> unresolved.
  - "I'm a supervisor and have joined the chat; I've processed your refund and it reaches your account in 5-7 working days" -> resolved.
  - "I've escalated this to my supervisor, who will contact you tomorrow" -> unresolved.

## Sentiment
- Sentiment is the customer's emotional tone, not the topic. Calmly reporting a bad event is neutral.
- Negative needs expressed frustration, anger, worry or distress.
- Positive needs expressed satisfaction or relief beyond courtesy.
- Courtesy is neutral: "thanks", "ok", "bye", "have a nice day".
- Examples:
  - "My card was stolen yesterday; I'd like to block it." -> neutral.
  - "My card was stolen and I'm really worried someone is using it!" -> negative.
  - "Ok, thanks, bye." -> neutral.
  - "That's a huge relief, thank you so much!" -> positive.

## Escalation
- escalation_requested=true only if the CUSTOMER explicitly asks for a supervisor, a manager or escalation. An agent offering or mentioning escalation does not count. escalation_requested=false: no such customer request.

## Evidence
- Evidence is a list of ZERO-BASED turn indices and is never empty; cite only existing turns.
- Sentiment and escalation evidence cite customer turns only. start_sentiment evidence includes the first customer turn and end_sentiment evidence includes the last customer turn.
- For escalation_requested=false, cite the customer turns checked.

## Confusable intents
- top_up_by_card_charge: Whether, and how much, the customer pays to fund the account from a payment card, often one issued abroad.
- topping_up_by_card: How to fund the account from a payment card, or why a card top-up did not arrive; fee questions belong elsewhere.
- card_payment_fee_charged: An unexpected extra fee applied to a purchase paid with the card, not to a top-up or a cash withdrawal.
- top_up_by_bank_transfer_charge: Whether a fee applies when money is added to the account by an incoming bank transfer, including SEPA or SWIFT.
- get_physical_card: Finding, receiving or setting up the PIN that belongs to a newly issued physical card.
- order_physical_card: Ordering a physical card to be sent out, including how to request it and what it costs.
- getting_spare_card: Getting an additional card for the same account while keeping the current one, for oneself or a family member.
- change_pin: Changing a working card PIN to a new one, in the app or at a cash machine.
- pin_blocked: The card PIN no longer works or was locked after too many wrong attempts.
- lost_or_stolen_phone: Securing the account and regaining app access after the phone used for banking was lost or stolen.
- passcode_forgotten: Resetting a forgotten app login passcode while the customer still has their phone.
- lost_or_stolen_card: Reporting a physical card as lost or stolen so that it is blocked and replaced.
- request_refund: Asking to get money back for a purchase, or to cancel it, before any refund has been issued.
- Refund_not_showing_up: A merchant says a refund was issued, but the money has not appeared in the account.
- reverted_card_payment?: A card payment was reversed, cancelled or returned to the account, and the customer wants to know why.
