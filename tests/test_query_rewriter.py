"""test_query_rewriter.py — Heuristics and cleanup for query rewriting."""

from __future__ import annotations

import pytest

from app.query_rewriter import (
    _sanitize,
    build_rewrite_prompt,
    is_indic_or_hinglish,
    needs_rewrite,
    rewrite_query,
)


@pytest.fixture
def history() -> list[dict]:
    return [
        {"role": "user", "content": "What are the rules for study leave?"},
        {"role": "assistant", "content": "Study leave can be requested by students who "
                                         "have completed academic activities for at least "
                                         "2 semesters."},
    ]


class TestNeedsRewrite:
    def test_tanpa_riwayat_tidak_perlu(self):
        assert needs_rewrite("What is the maximum?", []) is False
        assert needs_rewrite("What is the maximum?", None) is False

    def test_riwayat_tanpa_jawaban_asisten_tidak_perlu(self):
        """Nothing to refer to yet."""
        hanya_user = [{"role": "user", "content": "Hello"}]
        assert needs_rewrite("What is the maximum?", hanya_user) is False

    @pytest.mark.parametrize("query", [
        "What is the maximum?",
        "What is that?",
        "How long maximum?",
        "What is the procedure?",
    ])
    def test_pertanyaan_pendek_perlu_ditulis_ulang(self, query, history):
        assert needs_rewrite(query, history) is True

    @pytest.mark.parametrize("query", [
        "How about the fee?",
        "What are the requirements for that?",
        "How about the regulations for master's programs?",
    ])
    def test_kata_rujukan_perlu_ditulis_ulang(self, query, history):
        assert needs_rewrite(query, history) is True

    @pytest.mark.parametrize("query", [
        "What is the minimum GPA to graduate cum laude?",
        "What are the re-registration requirements for new students?",
        "What is the minimum total credits to graduate bachelor program?",
    ])
    def test_pertanyaan_mandiri_tidak_perlu(self, query, history):
        assert needs_rewrite(query, history) is False

    def test_query_kosong(self, history):
        assert needs_rewrite("", history) is False
        assert needs_rewrite("   ", history) is False

    @pytest.mark.parametrize("query", [
        "सीएसई की ओपनिंग रैंक क्या है?",
        "होस्टल के फीस कतेक लागही?",
        "सीएसई म एडमिशन कइसे मिलही?",
        "छात्रवृत्ति बर का नियम हावे?",
    ])
    def test_devanagari_perlu_tanpa_riwayat(self, query):
        """Devanagari Hindi and Chhattisgarhi must trigger rewrite on Turn 1."""
        assert needs_rewrite(query, None) is True
        assert needs_rewrite(query, []) is True

    @pytest.mark.parametrize("query", [
        "CSE ki opening rank kya hai",
        "Hostel ki fees kitni hai",
        "MTech me admission kaise milega",
        "Scholarship ke liye eligibility criteria kya hai",
        "Placement kaisa hai CSE ka",
    ])
    def test_hinglish_perlu_tanpa_riwayat(self, query):
        """Hinglish must trigger rewrite on Turn 1 so search gets English query."""
        assert needs_rewrite(query, None) is True
        assert needs_rewrite(query, []) is True

    @pytest.mark.parametrize("query", [
        "Okhar fees katek hawe",
        "cse me admission kaise milhi",
        "hostel kisan hawe",
        "branch sliding bar kaabar niyam hawe",
    ])
    def test_chhattisgarhi_perlu_tanpa_riwayat(self, query):
        """Romanized Chhattisgarhi must trigger rewrite on Turn 1."""
        assert needs_rewrite(query, None) is True
        assert needs_rewrite(query, []) is True

    @pytest.mark.parametrize("query", [
        "Iska cutoff kitna gaya tha?",
        "Okhar eligibility batawa",
        "Aur hostel charge kitna lagega?",
        "Usme scholarship milegi?",
    ])
    def test_kata_rujukan_indic_perlu_ditulis_ulang(self, query, history):
        """Indic referential follow-ups must be rewritten."""
        assert needs_rewrite(query, history) is True


class TestBuildRewritePrompt:
    def test_memuat_riwayat_dan_pertanyaan(self, history):
        prompt = build_rewrite_prompt("What is the maximum?", history)
        assert "study leave" in prompt
        assert "What is the maximum?" in prompt

    def test_peran_diberi_label(self, history):
        prompt = build_rewrite_prompt("q", history)
        assert "User:" in prompt
        assert "Assistant:" in prompt

    def test_pesan_panjang_dipotong(self):
        panjang = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
        ]
        prompt = build_rewrite_prompt("q", panjang)
        assert len(prompt) < 2000

    def test_hanya_jendela_terakhir_dipakai(self):
        banyak = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"message-{i}"}
            for i in range(20)
        ]
        prompt = build_rewrite_prompt("q", banyak)
        assert "message-19" in prompt
        assert "message-0" not in prompt


class TestSanitize:
    def test_teks_bersih_diteruskan(self):
        assert _sanitize("What is the maximum study leave?", "original") == "What is the maximum study leave?"

    def test_tanda_kutip_dibuang(self):
        assert _sanitize('"What is the maximum leave?"', "original") == "What is the maximum leave?"

    def test_awalan_dibuang(self):
        assert _sanitize("Standalone question: What is the maximum leave?", "original") == \
               "What is the maximum leave?"
        assert _sanitize("Standalone English Question: What is the cutoff?", "original") == \
               "What is the cutoff?"
        assert _sanitize("Rewritten query: What are the fees?", "original") == \
               "What are the fees?"

    def test_markdown_bold_dibuang(self):
        assert _sanitize("**What is the cutoff rank for CSE?**", "original") == \
               "What is the cutoff rank for CSE?"

    def test_hanya_baris_pertama_dipakai(self):
        hasil = _sanitize("What is the maximum leave?\nAdditional explanation.", "original")
        assert hasil == "What is the maximum leave?"

    def test_balasan_kosong_jatuh_ke_asli(self):
        assert _sanitize("", "original query") == "original query"
        assert _sanitize("   ", "original query") == "original query"

    def test_balasan_terlalu_panjang_ditolak(self):
        assert _sanitize("explanation " * 200, "original query") == "original query"


class TestRewriteQuery:
    def test_dilewati_saat_heuristik_bilang_tidak_perlu(self, history, monkeypatch):
        def jangan_dipanggil(*args, **kwargs):
            raise AssertionError("LLM should not be called")

        monkeypatch.setattr("app.query_rewriter.call_utility_llm", jangan_dipanggil)
        mandiri = "What is the minimum GPA to graduate cum laude?"
        assert rewrite_query(mandiri, history) == mandiri

    def test_force_melewati_heuristik(self, history, monkeypatch):
        monkeypatch.setattr(
            "app.query_rewriter.call_utility_llm",
            lambda *a, **k: "Rewritten question?",
        )
        hasil = rewrite_query("A fairly long standalone question?",
                              history, force=True)
        assert hasil == "Rewritten question?"

    def test_menulis_ulang_pertanyaan_lanjutan(self, history, monkeypatch):
        monkeypatch.setattr(
            "app.query_rewriter.call_utility_llm",
            lambda *a, **k: "What is the maximum study leave that can be requested?",
        )
        hasil = rewrite_query("What is the maximum?", history)
        assert hasil == "What is the maximum study leave that can be requested?"

    def test_kegagalan_api_jatuh_ke_query_asli(self, history, monkeypatch):
        monkeypatch.setattr(
            "app.query_rewriter.call_utility_llm",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API down")),
        )
        assert rewrite_query("What is the maximum?", history) == "What is the maximum?"

    def test_tanpa_riwayat_mengembalikan_apa_adanya(self):
        assert rewrite_query("What is the maximum?", None) == "What is the maximum?"

    def test_hinglish_turn_1_diterjemahkan_tanpa_riwayat(self, monkeypatch):
        monkeypatch.setattr(
            "app.query_rewriter.call_utility_llm",
            lambda *a, **k: "What is the opening rank for B.Tech Computer Science and Engineering?",
        )
        hasil = rewrite_query("CSE ki opening rank kya hai", None)
        assert hasil == "What is the opening rank for B.Tech Computer Science and Engineering?"

    def test_devanagari_turn_1_diterjemahkan_tanpa_riwayat(self, monkeypatch):
        monkeypatch.setattr(
            "app.query_rewriter.call_utility_llm",
            lambda *a, **k: "What is the fee structure for hostel accommodation?",
        )
        hasil = rewrite_query("होस्टल के फीस कतेक लागही?", None)
        assert hasil == "What is the fee structure for hostel accommodation?"
