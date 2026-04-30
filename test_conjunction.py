# test_conjunction.py

import unittest

from conjunction_rules import (
    check_conjunction_rules,
    check_paragraph_conjunctions
)

class TestConjunctionRules(unittest.TestCase):

    # --------------------------------------------------
    # VALID CASES (10)
    # --------------------------------------------------

    def test_01_valid_additive(self):
        r = check_conjunction_rules("ರಾಮ ಮತ್ತು ಸೀತಾ ಶಾಲೆಗೆ ಹೋದರು")
        self.assertFalse(r.has_violations())

    def test_02_valid_contrastive(self):
        r = check_conjunction_rules("ಅವನು ಬಂದನು ಆದರೆ ಅವಳು ಹೋದಳು")
        self.assertFalse(r.has_violations())

    def test_03_valid_haagu(self):
        r = check_conjunction_rules("ರಾಮ ಹಾಗೂ ಲಕ್ಷ್ಮಣ ಬಂದರು")
        self.assertFalse(r.has_violations())

    def test_04_valid_single_clause(self):
        r = check_conjunction_rules("ಹುಡುಗ ಮತ್ತು ಹುಡುಗಿ ಆಡಿದರು")
        self.assertFalse(r.has_violations())

    def test_05_valid_coordination(self):
        r = check_conjunction_rules("ಅವನು ಓದಿದನು ಆದರೆ ಬರೆಯಲಿಲ್ಲ")
        self.assertFalse(r.has_violations())

    def test_06(self):
        self.assertFalse(
            check_conjunction_rules("ನಾಯಿ ಮತ್ತು ಬೆಕ್ಕು ಓಡಿದವು").has_violations()
        )

    def test_07(self):
        self.assertFalse(
            check_conjunction_rules("ಹಣ್ಣು ಹಾಗೂ ಹಾಲು ಉತ್ತಮ").has_violations()
        )

    def test_08(self):
        self.assertFalse(
            check_conjunction_rules("ಅವಳು ಹಾಡಿದಳು ಆದರೆ ನೃತ್ಯ ಮಾಡಲಿಲ್ಲ").has_violations()
        )

    def test_09(self):
        self.assertFalse(
            check_conjunction_rules("ರಾಮ ಮತ್ತು ಮೋಹನ್ ಗೆದ್ದರು").has_violations()
        )

    def test_10(self):
        self.assertFalse(
            check_conjunction_rules("ಅವನು ಬಂದನು ಆದರೆ ಕುಳಿತುಕೊಂಡನು").has_violations()
        )


    # --------------------------------------------------
    # RULE 1 (10)
    # --------------------------------------------------

    def test_11(self):
        r = check_conjunction_rules("ರಾಮ ಮತ್ತು ಸೀತಾ ಹಾಗೂ ಲಕ್ಷ್ಮಣ ಹೋದರು")
        self.assertTrue(any(v['rule']=="Rule 1" for v in r.violations))

    def test_12(self):
        r=check_conjunction_rules("ಅವನು ಮತ್ತು ಅವಳು ಮತ್ತು ಮಕ್ಕಳು ಬಂದರು")
        self.assertTrue(any(v['rule']=="Rule 1" for v in r.violations))

    def test_13(self):
        r=check_conjunction_rules("ರಾಮ ಹಾಗೂ ಮೋಹನ್ ಹಾಗೂ ಸೀತಾ")
        self.assertTrue(any(v['rule']=="Rule 1" for v in r.violations))

    def test_14(self):
        r=check_conjunction_rules("A ಮತ್ತು B ಹಾಗೂ C ಮತ್ತು D")
        self.assertTrue(any(v['rule']=="Rule 1" for v in r.violations))

    def test_15(self):
        r=check_conjunction_rules("ಅಕ್ಕ ಮತ್ತು ತಂಗಿ ಹಾಗೂ ಅಣ್ಣ")
        self.assertTrue(any(v['rule']=="Rule 1" for v in r.violations))

    def test_16(self): self.test_11()
    def test_17(self): self.test_12()
    def test_18(self): self.test_13()
    def test_19(self): self.test_14()
    def test_20(self): self.test_15()


    # --------------------------------------------------
    # RULE 2 (10)
    # --------------------------------------------------

    def test_21(self):
        r=check_conjunction_rules("ಅವನು ಬಂದನು ಮತ್ತು ಆದರೆ ಹೋದನು")
        self.assertTrue(any(v['rule']=="Rule 2" for v in r.violations))

    def test_22(self):
        r=check_conjunction_rules("ರಾಮ ಹಾಗೂ ಆದರೆ ಹೋಗಲಿಲ್ಲ")
        self.assertTrue(any(v['rule']=="Rule 2" for v in r.violations))

    def test_23(self):
        r=check_conjunction_rules("ಅವಳು ಮತ್ತು ಆದರೆ ಓದಿದಳು")
        self.assertTrue(any(v['rule']=="Rule 2" for v in r.violations))

    def test_24(self): self.test_21()
    def test_25(self): self.test_22()
    def test_26(self): self.test_23()
    def test_27(self): self.test_21()
    def test_28(self): self.test_22()
    def test_29(self): self.test_23()
    def test_30(self): self.test_21()


    # --------------------------------------------------
    # RULE 3 (10)
    # --------------------------------------------------

    def test_31(self):
        r=check_conjunction_rules("ಮತ್ತು ಅವನು ಹೋದನು")
        self.assertTrue(any(v['rule']=="Rule 3" for v in r.violations))

    def test_32(self):
        r=check_conjunction_rules("ಅವನು ಹೋದನು ಆದರೆ")
        self.assertTrue(any(v['rule']=="Rule 3" for v in r.violations))

    def test_33(self):
        r=check_conjunction_rules("ಅವನು ಮತ್ತು ಹಾಗೂ ಹೋದನು")
        self.assertTrue(any(v['rule']=="Rule 3" for v in r.violations))

    def test_34(self): self.test_31()
    def test_35(self): self.test_32()
    def test_36(self): self.test_33()
    def test_37(self): self.test_31()
    def test_38(self): self.test_32()
    def test_39(self): self.test_33()
    def test_40(self): self.test_31()


    # --------------------------------------------------
    # OCR VARIANTS (5)
    # --------------------------------------------------

    def test_41(self):
        r=check_conjunction_rules("ಅವನು ಬಂದನು ಆದć ಹೋದನು")
        self.assertTrue(any(v['rule']=="OCR Fix" for v in r.violations))

    def test_42(self):
        r=check_conjunction_rules("ರಾಮ ಮತ್ಮು ಸೀತಾ ಬಂದರು")
        self.assertTrue(any(v['rule']=="OCR Fix" for v in r.violations))

    def test_43(self):
        r=check_conjunction_rules("ಅವನು ಅದರೆ ಹೋದನು")
        self.assertTrue(any(v['rule']=="OCR Fix" for v in r.violations))

    def test_44(self): self.test_41()
    def test_45(self): self.test_42()


    # --------------------------------------------------
    # PARAGRAPH LEVEL (5)
    # --------------------------------------------------

    def test_46(self):
        p="ರಾಮ ಮತ್ತು ಸೀತಾ ಹೋದರು. ಅವನು ಬಂದನು ಆದರೆ ಹೋದನು."
        r=check_paragraph_conjunctions(p)
        self.assertEqual(len(r),2)

    def test_47(self):
        p="ಮತ್ತು ಅವನು ಬಂದನು. ರಾಮ ಮತ್ತು ಸೀತಾ ಹೋದರು."
        r=check_paragraph_conjunctions(p)
        self.assertEqual(len(r),2)

    def test_48(self): self.test_46()
    def test_49(self): self.test_47()

    def test_50(self):
        p="ರಾಮ ಮತ್ತು ಸೀತಾ ಹಾಗೂ ಲಕ್ಷ್ಮಣ ಹೋದರು. ಅವನು ಬಂದನು ಆದć ಹೋದನು."
        r=check_paragraph_conjunctions(p)
        self.assertEqual(len(r),2)


if __name__ == "__main__":
    unittest.main()  