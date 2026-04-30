from conjunction_rules import check_conjunction_rules

test_cases = [
    {
        "id":1,
        "input":"ರಾಮ ಮತ್ತು ಸೀತಾ ಹಾಗೂ ಲಕ್ಷ್ಮಣ ಹೋದರು",
        "expected":"Rule 1"
    },
    {
        "id":2,
        "input":"ಅವನು ಬಂದನು ಮತ್ತು ಆದರೆ ಹೋದನು",
        "expected":"Rule 2"
    },
    {
        "id":3,
        "input":"ಮತ್ತು ಅವಳು ಶಾಲೆಗೆ ಹೋದಳು",
        "expected":"Rule 3"
    },
    {
        "id":4,
        "input":"ಅವನು ಊಟ ಮಾಡಿದನು ಆದರೆ",
        "expected":"Rule 3"
    },
    {
        "id":5,
        "input":"ಅವನು ಬಂದನು ಆದć ಹೋದನು",
        "expected":"OCR Fix"
    },
    {
        "id":6,
        "input":"ರಾಮ ಮತ್ತು ಸೀತಾ ಶಾಲೆಗೆ ಹೋದರು",
        "expected":"No violation"
    }
]

passed = 0

print("="*70)
print("CONJUNCTION RULE TEST RESULTS")
print("="*70)

for tc in test_cases:

    r = check_conjunction_rules(tc["input"])

    if r.has_violations():
        obtained = ", ".join(
            [v['rule'] for v in r.violations]
        )
    else:
        obtained = "No violation"

    status = "PASS" if tc["expected"] in obtained else "FAIL"

    if status=="PASS":
        passed += 1

    print("\n"+"-"*70)
    print(f"Test Case : {tc['id']}")
    print(f"Input     : {tc['input']}")
    print(f"Expected  : {tc['expected']}")
    print(f"Obtained  : {obtained}")
    print(f"Status    : {status}")

    if r.correction_made:
        print(f"Corrected : {r.corrected_sentence}")

print("\n"+"="*70)
print(f"Passed {passed}/{len(test_cases)}")
print("="*70)