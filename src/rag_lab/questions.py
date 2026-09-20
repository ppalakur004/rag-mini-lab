REQUIRED_QUESTIONS = [
    {
        "question": "How much can I spend on food each day?",
        "expected_section": "1. Meals",
        "must_refuse": False,
        "answer_contains": ["$65"],
    },
    {
        "question": "Can I book first-class airfare?",
        "expected_section": "3. Airfare",
        "must_refuse": False,
        "answer_contains": ["economy"],
    },
    {
        "question": "My hotel costs $250. What do I need?",
        "expected_section": "2. Hotels",
        "must_refuse": False,
        "answer_contains": ["manager"],
    },
    {
        "question": "Do I need a receipt for a $20 taxi?",
        "expected_section": "5. Receipts",
        "must_refuse": False,
        "answer_contains": ["25"],
    },
    {
        "question": "Can I claim a limousine upgrade?",
        "expected_section": "4. Ground Transportation",
        "must_refuse": False,
        "answer_contains": ["not reimbursable"],
    },
    {
        "question": "Does the company reimburse gym memberships?",
        "expected_section": None,
        "must_refuse": True,
        "answer_contains": ["The provided policy does not answer this question."],
    },
]
