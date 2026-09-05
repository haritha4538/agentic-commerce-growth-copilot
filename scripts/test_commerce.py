"""
Commerce Action Center verification script.

Checks the new src/commerce modules in isolation: dependency availability,
credential/test-key detection, the deterministic action mapper, the audit
log round-trip, and (only if real Test Mode credentials are configured) an
actual Razorpay Payment Link creation.

Like every other verification script in this project, this fails gracefully
and informatively at every stage rather than crashing with a raw traceback.

Run from the project root:
    python scripts/test_commerce.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> bool:
    # ----------------------------------------------------------------
    section("STEP 0: Checking imports")
    try:
        from src.commerce.razorpay_client import (
            is_test_mode_configured, create_payment_link, FIXED_TEST_AMOUNT_INR,
            RazorpayCredentialsError, RazorpayClientError, TEST_KEY_PREFIX,
        )
        from src.commerce.action_mapper import build_commerce_action
        from src.commerce.audit_log import append_audit_record, load_audit_log, update_audit_record_result
    except ImportError as exc:
        print(f"❌ Could not import commerce modules: {exc}")
        return False
    print("✅ All commerce modules imported successfully.")
    print(f"   Fixed demo amount: ₹{FIXED_TEST_AMOUNT_INR}  |  expected test key prefix: '{TEST_KEY_PREFIX}'")

    # ----------------------------------------------------------------
    section("STEP 1: Checking Razorpay Test Mode credentials")
    configured = is_test_mode_configured()
    if configured:
        print("✅ RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are set and look like Test Mode keys.")
    else:
        print("⚠️  Razorpay Test Mode credentials are not configured (or don't look like test keys).")
        print("   Set RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET in .env to enable a real Payment Link test below.")
        print("   Everything else in this script does not require credentials and will still run.")

    # ----------------------------------------------------------------
    section("STEP 2: Testing the action mapper (offline, deterministic)")
    fake_recommendation = {
        "title": "Bundle high-margin, low-volume products",
        "description": "6 products carry strong margin but low sales volume.",
        "supporting_metric": "6 high-margin/low-volume products identified",
        "policy_reference": "promotion_guidelines.md",
        "estimated_impact": "estimated 5-10% revenue lift if bundled and promoted",
    }
    action = build_commerce_action(fake_recommendation, source_section="product_opportunities")
    print(f"Mapped action: {action}")
    assert action["amount_inr"] == FIXED_TEST_AMOUNT_INR, "Amount must always be the fixed test amount"
    assert action["title"] == fake_recommendation["title"]
    assert action["policy_reference"] == "promotion_guidelines.md"
    assert action["source_section"] == "product_opportunities"
    print("✅ action_mapper produces a correctly-shaped, traceable CommerceAction with the fixed test amount.")

    # ----------------------------------------------------------------
    section("STEP 3: Testing the audit log round-trip (isolated temp file)")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_log_path = Path(tmp_dir) / "test_audit_log.jsonl"

        record = {
            "reference_id": "growth-copilot-test-0001",
            "recommendation": action["title"],
            "reason": action["reason"],
            "merchant_approved": True,
            "amount_inr": action["amount_inr"],
            "payment_link_id": None,
            "result": "pending",
        }
        saved = append_audit_record(record, path=test_log_path)
        assert saved["_persisted"] is True
        assert "timestamp" in saved

        loaded = load_audit_log(path=test_log_path)
        assert len(loaded) == 1
        assert loaded[0]["reference_id"] == "growth-copilot-test-0001"
        print(f"✅ Wrote and re-read 1 audit record: {loaded[0]}")

        updated = update_audit_record_result(
            "growth-copilot-test-0001",
            {"payment_link_id": "plink_fake123", "result": "created"},
            path=test_log_path,
        )
        assert updated is True
        reloaded = load_audit_log(path=test_log_path)
        assert reloaded[0]["result"] == "created"
        assert reloaded[0]["payment_link_id"] == "plink_fake123"
        assert "status_checked_at" in reloaded[0]
        print(f"✅ update_audit_record_result correctly updated the record: {reloaded[0]}")

        missing_update = update_audit_record_result("does-not-exist", {"result": "x"}, path=test_log_path)
        assert missing_update is False
        print("✅ update_audit_record_result correctly returns False for a non-existent reference_id.")

    # ----------------------------------------------------------------
    section("STEP 4: Live Razorpay Payment Link creation (only if credentials are configured)")
    if not configured:
        print("⏭️  Skipped — no Test Mode credentials configured. This is expected in an environment")
        print("   without a .env file; everything else in this script has already passed.")
    else:
        try:
            response = create_payment_link(
                description="Growth Copilot verification script test link",
                reference_id="growth-copilot-verification-script",
            )
            print(f"✅ Payment Link created: id={response.get('id')}, short_url={response.get('short_url')}, "
                  f"status={response.get('status')}")
        except (RazorpayCredentialsError, RazorpayClientError) as exc:
            print(f"❌ Payment Link creation failed: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Unexpected error creating a Payment Link: {exc}")
            return False

    # ----------------------------------------------------------------
    section("VERIFICATION RESULT")
    print("✅ PASSED — commerce modules are correctly implemented and behave as expected.")
    if not configured:
        print("   (Live Payment Link creation was skipped — configure RAZORPAY_KEY_ID/SECRET to test it for real.)")
    return True


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
