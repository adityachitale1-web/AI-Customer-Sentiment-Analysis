"""Populate feedback.db with realistic sample data so the dashboard has
something to show before (or in addition to) live API traffic.

Usage:  python seed_sample_data.py
"""

import random
from datetime import datetime, timedelta

from db import init_db, insert_feedback, sentiment_counts

random.seed(42)

SAMPLES = {
    "Positive": [
        "Absolutely love the new update, everything feels faster!",
        "Customer support resolved my issue in under five minutes. Impressive.",
        "Great value for money, would definitely recommend to friends.",
        "The delivery arrived a day early and packaging was perfect.",
        "Best purchase I've made this year, quality is outstanding.",
        "The app is so intuitive, my parents figured it out instantly.",
        "Five stars — the checkout process is smooth and painless.",
        "Really happy with the battery life, lasts me two full days.",
        "The onboarding tutorial was clear and genuinely helpful.",
        "Refund was processed the same day I asked. Amazing service.",
        "Love the dark mode, thank you for finally adding it!",
        "Product matches the description exactly. Very satisfied.",
    ],
    "Negative": [
        "App keeps crashing every time I open the settings page.",
        "Waited 45 minutes on hold and then got disconnected. Unacceptable.",
        "The product broke after two days of normal use.",
        "Delivery was a week late and nobody responded to my emails.",
        "Latest update removed the one feature I actually used.",
        "Charged twice for the same order and still waiting for a refund.",
        "The quality has really gone downhill compared to last year.",
        "Website checkout fails with an error at the payment step.",
        "Instructions are confusing and the support articles are outdated.",
        "Way too expensive for what you actually get.",
        "The item arrived damaged and the return process is a nightmare.",
        "Notifications are constant spam, no way to turn them off.",
    ],
    "Neutral": [
        "Received the package today, haven't opened it yet.",
        "Does this model come in other colors?",
        "The product is okay, nothing special but it works.",
        "Average experience overall, about what I expected.",
        "I've been using it for a week, still forming an opinion.",
        "How do I export my data to a spreadsheet?",
        "It works as described. Standard stuff.",
        "The store was moderately busy when I visited.",
        "Switched over from a competitor, comparing features now.",
        "Is there a student discount available?",
        "Setup took about twenty minutes, more or less as expected.",
        "The manual says to update firmware first, doing that now.",
    ],
}

SOURCES = ["web_form", "app_review", "email", "twitter", "survey"]


def main() -> None:
    init_db()
    now = datetime.now()
    inserted = 0
    for sentiment, texts in SAMPLES.items():
        for text in texts:
            # Spread entries over the past 30 days so the trend chart has shape
            days_ago = random.uniform(0, 30)
            created = (now - timedelta(days=days_ago)).isoformat(timespec="seconds")
            confidence = round(random.uniform(0.72, 0.99), 4)
            insert_feedback(text, sentiment, confidence,
                            source=random.choice(SOURCES), created_at=created)
            inserted += 1
    print(f"Inserted {inserted} sample rows.")
    print("Counts by sentiment:", sentiment_counts())


if __name__ == "__main__":
    main()
