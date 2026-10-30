"""Test public circuit breaker for invalid provider responses."""

from app.answer_guard import diagnose_answer, has_page_citation, is_safe_answer


class TestAnswerGuard:
    def test_jawaban_dengan_sitasi_lolos(self):
        answer = "Maximum study leave is 2 semesters. 📄 Source: Page 40 — Section Title"
        assert has_page_citation(answer) is True
        assert is_safe_answer(answer) is True

    def test_sitasi_huruf_kecil_lolos(self):
        assert has_page_citation("📄 source: page 40") is True

    def test_kalimat_motivasi_provider_diblokir(self):
        assert is_safe_answer("Success starts with a single decision.") is False

    def test_pesan_error_diblokir(self):
        assert is_safe_answer("⚠️ Usage limit reached.") is False
        assert is_safe_answer("404 NOT_FOUND model") is False

    def test_jawaban_kosong_diblokir(self):
        assert is_safe_answer("") is False
        assert is_safe_answer("   ") is False

    def test_jawaban_tanpa_sitasi_diblokir(self):
        assert is_safe_answer("Study leave is max 2 semesters.") is False

    def test_diagnosis_provider_rate_limit_tidak_membocorkan_error_mentah(self):
        diagnosis = diagnose_answer(
            "⚠️ Rate limit reached. Do not display secret-token",
            {39, 40, 41},
        )
        assert diagnosis == (
            "provider_rate_limit",
            "Provider rejected request due to quota or rate limit (HTTP 429).",
        )
        assert "secret-token" not in diagnosis[1]

    def test_diagnosis_sitasi_hilang(self):
        diagnosis = diagnose_answer("Study leave is max 2 semesters.", {39, 40, 41})
        assert diagnosis == (
            "missing_page_citation",
            "Answer does not contain a citation in 'Page N' format.",
        )

    def test_diagnosis_halaman_sitasi_di_luar_konteks(self):
        diagnosis = diagnose_answer("Study leave is max 2 semesters. Page 43", {39, 40, 41})
        assert diagnosis is not None
        assert diagnosis[0] == "citation_outside_context"
        assert "43" in diagnosis[1]
        assert "39, 40, 41" in diagnosis[1]

    def test_diagnosis_jawaban_valid_tidak_menolak(self):
        answer = "Maximum leave is 2 semesters. 📄 Source: Page 40 — Leave Title"
        assert diagnose_answer(answer, {39, 40, 41}) is None

    def test_diagnosis_semua_model_provider_tidak_tersedia(self):
        diagnosis = diagnose_answer(
            '⚠️ An error occurred: 403 {"code":"model_disabled"}',
            {39, 40, 41},
        )
        assert diagnosis is not None
        assert diagnosis[0] == "provider_models_unavailable"
        assert "disabled or unavailable" in diagnosis[1]
