"""CUNCA-Hybrid tests.

Covers:
  TC.1  CUNCAConfig — GQA, sliding-window, ssm_ratio, presets
  TC.2  Architecture — CUNCAModel block layout, forward pass
  TC.3  QLoRA — NF4 quantization, LoRALinear, apply_qlora
  TC.4  CUNCA-Bench — tasks, scorers, CUNCABench evaluator
  TC.5  Energy profiler — EnergyProfiler, wall-clock fallback
  TC.6  CUNCA Corpus — C4Filter, MinHashLSH, CUNCACorpusProcessor
  TC.7  Sector demos — admin, industry, health FastAPI routes
  TC.8  Mix config — cunca_pretraining.yaml
"""
from __future__ import annotations

import pytest
from pathlib import Path

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

needs_torch = pytest.mark.skipif(not _TORCH, reason="torch not installed")

_MIX_CONFIGS = Path(__file__).parents[2] / "data" / "mix_configs"


# ===========================================================================
# TC.1 — CUNCAConfig
# ===========================================================================

class TestCUNCAConfig:
    def test_default_config(self):
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig()
        assert cfg.num_heads == 16
        assert cfg.num_kv_heads == 4
        assert cfg.kv_groups == 4
        assert cfg.window_size == 512
        assert cfg.ssm_ratio == 0.5

    def test_preset_1_3b(self):
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig.preset("1.3b")
        assert cfg.hidden_size == 2048
        assert cfg.num_layers == 24
        assert cfg.num_kv_heads == 4
        assert cfg.kv_groups == 4

    def test_preset_3b(self):
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig.preset("3b")
        assert cfg.hidden_size == 2560
        assert cfg.num_layers == 32
        assert cfg.num_kv_heads == 4

    def test_preset_7b(self):
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig.preset("7b")
        assert cfg.hidden_size == 4096
        assert cfg.num_kv_heads == 8
        assert cfg.kv_groups == 4

    def test_invalid_preset(self):
        from cunca.config import CUNCAConfig
        with pytest.raises(ValueError, match="Unknown preset"):
            CUNCAConfig.preset("999b")

    def test_kv_heads_must_divide_heads(self):
        from cunca.config import CUNCAConfig
        with pytest.raises(ValueError, match="divisible"):
            CUNCAConfig(num_heads=16, num_kv_heads=3)

    def test_ssm_ratio_out_of_range(self):
        from cunca.config import CUNCAConfig
        with pytest.raises(ValueError, match="ssm_ratio"):
            CUNCAConfig(ssm_ratio=1.5)

    def test_block_types_alternating(self):
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig(num_layers=4, ssm_ratio=0.5)
        types = cfg.block_types()
        assert len(types) == 4
        assert "attn" in types
        assert "ssm" in types

    def test_block_types_all_attn(self):
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig(num_layers=4, ssm_ratio=0.0)
        types = cfg.block_types()
        assert all(t == "attn" for t in types)

    def test_block_types_all_ssm(self):
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig(num_layers=4, ssm_ratio=1.0)
        types = cfg.block_types()
        assert all(t == "ssm" for t in types)

    def test_estimate_params_1_3b(self):
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig.preset("1.3b")
        params = cfg.estimate_params()
        # Rough check: should be in hundreds of millions
        assert params > 100_000_000


# ===========================================================================
# TC.2 — Architecture
# ===========================================================================

class TestCUNCAArchitecture:
    def test_import_without_torch(self):
        """Config imports must not require torch."""
        from cunca.config import CUNCAConfig
        cfg = CUNCAConfig.preset("1.3b")
        assert cfg.num_layers == 24

    @needs_torch
    def test_model_builds_1_3b(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAModel
        cfg = CUNCAConfig(
            hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2,
            intermediate_size=128, vocab_size=100, max_seq_len=32,
            window_size=8, ssm_ratio=0.5,
        )
        model = CUNCAModel(cfg)
        assert model.num_params() > 0

    @needs_torch
    def test_forward_pass(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAModel
        cfg = CUNCAConfig(
            hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=128, vocab_size=100, max_seq_len=16,
            window_size=8, ssm_ratio=0.5,
        )
        model = CUNCAModel(cfg)
        ids = torch.randint(0, 100, (2, 8))
        out = model(ids)
        assert out.shape == (2, 8, 100)

    @needs_torch
    def test_block_layout_alternating(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAModel, CUNCATransformerBlock, CUNCAMambaBlock
        cfg = CUNCAConfig(
            hidden_size=32, num_layers=4, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=16, ssm_ratio=0.5,
        )
        model = CUNCAModel(cfg)
        types = [type(b).__name__ for b in model.blocks]
        assert "CUNCATransformerBlock" in types
        assert "CUNCAMambaBlock" in types

    @needs_torch
    def test_all_transformer_blocks(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAModel, CUNCATransformerBlock
        cfg = CUNCAConfig(
            hidden_size=32, num_layers=4, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=16, ssm_ratio=0.0,
        )
        model = CUNCAModel(cfg)
        assert all(isinstance(b, CUNCATransformerBlock) for b in model.blocks)

    @needs_torch
    def test_gqa_kv_proj_size(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAAttention
        cfg = CUNCAConfig(
            hidden_size=64, num_layers=2, num_heads=8, num_kv_heads=2,
            intermediate_size=128, vocab_size=50, max_seq_len=16,
        )
        attn = CUNCAAttention(cfg)
        head_dim = cfg.hidden_size // cfg.num_heads
        kv_dim = cfg.num_kv_heads * head_dim
        assert attn.k_proj.out_features == kv_dim
        assert attn.v_proj.out_features == kv_dim
        assert attn.q_proj.out_features == cfg.hidden_size

    @needs_torch
    def test_block_summary(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAModel
        cfg = CUNCAConfig(
            hidden_size=32, num_layers=4, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=16, ssm_ratio=0.5,
        )
        model = CUNCAModel(cfg)
        summary = model.block_summary()
        assert len(summary) == 4
        assert all("type" in s and "params" in s for s in summary)


# ===========================================================================
# TC.3 — QLoRA
# ===========================================================================

class TestQLoRA:
    def test_nf4_quantize_round_trip(self):
        import numpy as np
        from cunca.qlora import nf4_quantize_weight, nf4_dequantize_weight
        w = np.random.randn(16, 32).astype(np.float32)
        codes, scales = nf4_quantize_weight(w)
        w_deq = nf4_dequantize_weight(codes, scales, w.shape)
        # Round-trip error should be small (within a few percent)
        rel_err = np.abs(w - w_deq).mean() / (np.abs(w).mean() + 1e-8)
        assert rel_err < 0.20, f"NF4 round-trip error too large: {rel_err:.4f}"

    def test_nf4_codes_in_range(self):
        import numpy as np
        from cunca.qlora import nf4_quantize_weight
        w = np.random.randn(8, 8).astype(np.float32)
        codes, scales = nf4_quantize_weight(w)
        assert codes.dtype == np.uint8
        assert codes.min() >= 0
        assert codes.max() <= 15

    def test_nf4_scales_positive(self):
        import numpy as np
        from cunca.qlora import nf4_quantize_weight
        w = np.random.randn(8, 8).astype(np.float32)
        _, scales = nf4_quantize_weight(w)
        assert (scales > 0).all()

    @needs_torch
    def test_lora_linear_forward(self):
        from cunca.qlora import LoRALinear
        layer = LoRALinear(32, 64, rank=4, alpha=8.0)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == (2, 10, 64)

    @needs_torch
    def test_lora_linear_load_base(self):
        from cunca.qlora import LoRALinear
        layer = LoRALinear(32, 64, rank=4, alpha=8.0)
        base_w = torch.randn(64, 32)
        layer.load_base_weight(base_w)
        assert layer._nf4_codes is not None

    @needs_torch
    def test_apply_qlora_replaces_layers(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAModel
        from cunca.qlora import apply_qlora, LoRALinear
        cfg = CUNCAConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=16, ssm_ratio=0.0,
        )
        model = CUNCAModel(cfg)
        replaced = apply_qlora(model, rank=4, alpha=8, target_modules=["q_proj", "v_proj"])
        assert replaced > 0
        # Check that some LoRALinear instances exist
        lora_modules = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
        assert len(lora_modules) == replaced

    @needs_torch
    def test_apply_qlora_freezes_base(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAModel
        from cunca.qlora import apply_qlora, qlora_stats
        cfg = CUNCAConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=16, ssm_ratio=0.0,
        )
        model = CUNCAModel(cfg)
        apply_qlora(model, rank=4, alpha=8, target_modules=["q_proj"])
        stats = qlora_stats(model)
        assert stats.trainable_params < stats.total_params
        assert stats.trainable_ratio < 0.1   # <10% trainable

    @needs_torch
    def test_qlora_stats(self):
        from cunca.config import CUNCAConfig
        from cunca.architecture import CUNCAModel
        from cunca.qlora import apply_qlora, qlora_stats
        cfg = CUNCAConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=16, ssm_ratio=0.0,
        )
        model = CUNCAModel(cfg)
        n_replaced = apply_qlora(model, rank=4, alpha=8, target_modules=["q_proj", "v_proj"])
        stats = qlora_stats(model)
        assert stats.replaced_layers == n_replaced
        assert stats.frozen_params + stats.trainable_params == stats.total_params


# ===========================================================================
# TC.4 — CUNCA-Bench
# ===========================================================================

class TestCUNCABench:
    def test_registry_has_12_tasks(self):
        from cunca.bench.tasks import REGISTRY
        assert len(REGISTRY) == 12

    def test_task_ids(self):
        from cunca.bench.tasks import REGISTRY
        expected = {f"T{i:02d}" for i in range(1, 13)}
        assert set(REGISTRY.keys()) == expected

    def test_list_tasks_all(self):
        from cunca.bench.tasks import list_tasks
        tasks = list_tasks()
        assert len(tasks) == 12

    def test_list_tasks_by_category(self):
        from cunca.bench.tasks import list_tasks
        comp = list_tasks("comprehension")
        assert len(comp) == 3
        trans = list_tasks("translation")
        assert len(trans) == 4
        reason = list_tasks("reasoning")
        assert len(reason) == 3
        safety = list_tasks("safety")
        assert len(safety) == 2

    def test_exact_match_scorer(self):
        from cunca.bench.tasks import REGISTRY
        task = REGISTRY["T01"]
        predictions = [s.reference for s in task.samples]
        result = task.evaluate(predictions)
        assert result.score == 1.0

    def test_exact_match_wrong_answers(self):
        from cunca.bench.tasks import REGISTRY
        task = REGISTRY["T01"]
        predictions = ["wrong answer"] * len(task.samples)
        result = task.evaluate(predictions)
        assert result.score == 0.0

    def test_bleu_scorer(self):
        from cunca.bench.tasks import REGISTRY
        task = REGISTRY["T04"]   # translation
        predictions = [s.reference for s in task.samples]
        result = task.evaluate(predictions)
        assert result.score > 0.0

    def test_safety_scorer_refusals(self):
        from cunca.bench.tasks import REGISTRY
        task = REGISTRY["T11"]
        # Simulate model refusing all → matches "refuse" references
        predictions = ["Non podo axudarte con iso." if s.reference == "refuse" else "Santiago de Compostela"
                       for s in task.samples]
        result = task.evaluate(predictions)
        assert result.score > 0.0

    def test_bench_evaluator_runs(self):
        from cunca.bench.evaluator import CUNCABench
        bench = CUNCABench(generate_fn=lambda p: "resposta de proba", tasks=["T01", "T08"])
        report = bench.run()
        assert len(report.results) == 2
        assert all(r.passed for r in report.results)

    def test_bench_report_by_category(self):
        from cunca.bench.evaluator import CUNCABench
        bench = CUNCABench(generate_fn=lambda p: "test")
        report = bench.run()
        cats = report.by_category()
        assert "comprehension" in cats
        assert "translation" in cats

    def test_bench_report_summary_string(self):
        from cunca.bench.evaluator import CUNCABench
        bench = CUNCABench(generate_fn=lambda p: "test", tasks=["T01"])
        report = bench.run()
        summary = report.summary()
        assert "CUNCA-Bench" in summary
        assert "T01" in summary

    def test_available_tasks(self):
        from cunca.bench.evaluator import CUNCABench
        tasks = CUNCABench.available_tasks()
        assert len(tasks) == 12


# ===========================================================================
# TC.5 — Energy profiler
# ===========================================================================

class TestEnergyProfiler:
    def test_profiler_initialises(self):
        from cunca.energy.profiler import EnergyProfiler
        p = EnergyProfiler()
        # gpu_available should be bool (True or False, depends on host)
        assert isinstance(p.gpu_available, bool)

    def test_measure_context_returns_result(self):
        from cunca.energy.profiler import EnergyProfiler
        p = EnergyProfiler()
        with p.measure() as ctx:
            pass   # nothing to measure
        result = ctx.result(n_tokens=10)
        assert result.n_tokens == 10
        assert result.wall_time_s >= 0

    def test_energy_result_joules_per_token(self):
        from cunca.energy.profiler import EnergyResult
        r = EnergyResult(total_joules=10.0, wall_time_s=1.0, n_tokens=100, n_samples=10, gpu_available=False)
        assert abs(r.joules_per_token - 0.1) < 1e-9

    def test_energy_result_str(self):
        from cunca.energy.profiler import EnergyResult
        r = EnergyResult(total_joules=0.0, wall_time_s=0.5, n_tokens=0, n_samples=0, gpu_available=False)
        s = str(r)
        assert "GPU unavailable" in s

    def test_energy_context_manager(self):
        from cunca.energy.profiler import energy_context
        with energy_context() as ctx:
            x = 1 + 1
        result = ctx.result(n_tokens=2)
        assert result.wall_time_s >= 0

    def test_profile_callable(self):
        from cunca.energy.profiler import EnergyProfiler
        p = EnergyProfiler()
        result = p.profile_callable(lambda: None, n_tokens=5)
        assert result.n_tokens == 5


# ===========================================================================
# TC.6 — CUNCA Corpus
# ===========================================================================

class TestCUNCACorpus:
    def test_c4_filter_passes_good_text(self):
        from data.cunca_corpus import C4Filter
        filt = C4Filter(min_words=5, lang="gl")
        text = "O raposo pardo salta sobre o can preguiceiro e todos os animais do bosque o observan con atención."
        passed, reason = filt.apply(text)
        assert passed, f"Expected pass, got reason: {reason}"

    def test_c4_filter_rejects_short(self):
        from data.cunca_corpus import C4Filter
        filt = C4Filter(min_words=50, lang="gl")
        passed, reason = filt.apply("curto")
        assert not passed
        assert "too_short" in reason

    def test_c4_filter_rejects_lorem(self):
        from data.cunca_corpus import C4Filter
        filt = C4Filter(min_words=5, lang="gl")
        text = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
        passed, reason = filt.apply(text)
        assert not passed
        assert "lorem" in reason

    def test_c4_filter_rejects_empty(self):
        from data.cunca_corpus import C4Filter
        filt = C4Filter(min_words=5, lang="gl")
        passed, reason = filt.apply("")
        assert not passed

    def test_minhash_lsh_deduplication(self):
        from data.cunca_corpus import MinHashLSH
        lsh = MinHashLSH()
        text = "O galego é a lingua de Galicia e falase tamén en certas zonas de Portugal e de Castela."
        # First time: not a duplicate
        assert not lsh.is_duplicate("doc1", text)
        # Near-identical: should be detected as duplicate
        text2 = "O galego é a lingua de Galicia e falase tamén en certas zonas de Portugal e de Castela!"
        assert lsh.is_duplicate("doc2", text2)

    def test_minhash_lsh_different_docs(self):
        from data.cunca_corpus import MinHashLSH
        lsh = MinHashLSH()
        t1 = "O galego é a lingua de Galicia falada por miles de persoas."
        t2 = "A computación cuántica é un campo emerxente da informática moderna."
        lsh.is_duplicate("d1", t1)
        assert not lsh.is_duplicate("d2", t2)

    def test_corpus_processor_filters(self, tmp_path):
        from data.cunca_corpus import CUNCACorpusProcessor
        # Create a file with one good doc and one too-short doc
        good = "O galego é a lingua de Galicia " * 10
        bad = "curto"
        (tmp_path / "test.txt").write_text(f"{good}\n\n{bad}\n", encoding="utf-8")

        proc = CUNCACorpusProcessor(output_dir=tmp_path / "out", lang="gl", min_words=5, dedup=False)
        stats = proc.process_file(tmp_path / "test.txt")
        assert stats.total_docs >= 2
        assert stats.passed_filter >= 1

    def test_corpus_processor_deduplication(self, tmp_path):
        from data.cunca_corpus import CUNCACorpusProcessor
        text = "O galego é a lingua propia de Galicia e " * 20
        # Write same doc twice
        (tmp_path / "dup.txt").write_text(f"{text}\n\n{text}\n", encoding="utf-8")

        proc = CUNCACorpusProcessor(output_dir=tmp_path / "out2", lang="gl", min_words=5, dedup=True)
        stats = proc.process_file(tmp_path / "dup.txt")
        assert stats.deduplicated >= 1


# ===========================================================================
# TC.7 — Sector demos
# ===========================================================================

class TestDemos:
    @pytest.fixture
    def admin_client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from cunca.demos.admin import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def industry_client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from cunca.demos.industry import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def health_client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from cunca.demos.health import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_admin_health(self, admin_client):
        r = admin_client.get("/demo/admin/health")
        assert r.status_code == 200
        assert r.json()["on_premises"] is True

    def test_admin_summarise(self, admin_client):
        r = admin_client.post("/demo/admin/summarise", json={
            "text": "Este é un documento oficial da Xunta de Galicia relativo ao procedemento administrativo.",
            "lang": "gl",
        })
        assert r.status_code == 200
        body = r.json()
        assert "summary" in body
        assert body["on_premises"] is True

    def test_admin_regulation(self, admin_client):
        r = admin_client.post("/demo/admin/regulation", json={
            "question": "Cales son os prazos para recorrer unha resolución administrativa?",
            "lang": "gl",
        })
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "disclaimer" in body

    def test_industry_health(self, industry_client):
        r = industry_client.get("/demo/industry/health")
        assert r.status_code == 200

    def test_industry_anomaly_low(self, industry_client):
        r = industry_client.post("/demo/industry/anomaly", json={
            "sensor_id": "PUMP-A1",
            "metric": "vibration_hz",
            "value": 10.0,
            "threshold": 10.0,
        })
        assert r.status_code == 200
        assert r.json()["severity"] == "low"

    def test_industry_anomaly_critical(self, industry_client):
        r = industry_client.post("/demo/industry/anomaly", json={
            "sensor_id": "MOTOR-B3",
            "metric": "temperature_c",
            "value": 250.0,
            "threshold": 80.0,
        })
        assert r.status_code == 200
        assert r.json()["severity"] == "critical"

    def test_industry_workorder(self, industry_client):
        r = industry_client.post("/demo/industry/workorder", json={
            "equipment_id": "CNC-01",
            "fault_description": "Ruído anormal no cabezal de fresado",
            "technician": "Xosé López",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["equipment_id"] == "CNC-01"
        assert body["order_id"].startswith("WO-")

    def test_health_demo_health(self, health_client):
        r = health_client.get("/demo/health/health")
        assert r.status_code == 200
        assert r.json()["on_premises"] is True

    def test_health_summarise(self, health_client):
        r = health_client.post("/demo/health/summarise", json={
            "note": "Paciente de 65 anos con hipertensión arterial e diabetes tipo 2 controladas. "
                    "Acude por control rutinario. Tensión 130/80. Glicemia 110 mg/dL.",
            "patient_id": "P-ANON-001",
        })
        assert r.status_code == 200
        body = r.json()
        assert "summary" in body
        assert "disclaimer" in body

    def test_health_icd10_hypertension(self, health_client):
        r = health_client.post("/demo/health/icd10", json={
            "diagnosis_text": "Hipertensión arterial esencial",
        })
        assert r.status_code == 200
        body = r.json()
        codes = [s["code"] for s in body["suggestions"]]
        assert "I10" in codes

    def test_health_icd10_unknown(self, health_client):
        r = health_client.post("/demo/health/icd10", json={
            "diagnosis_text": "Enfermidade descoñecida rara",
        })
        assert r.status_code == 200
        body = r.json()
        # Should return fallback
        assert len(body["suggestions"]) >= 1


# ===========================================================================
# TC.8 — CUNCA mix config
# ===========================================================================

class TestCUNCAMixConfig:
    def test_cunca_pretraining_yaml_exists(self):
        assert (_MIX_CONFIGS / "cunca_pretraining.yaml").exists()

    def test_cunca_pretraining_parses(self):
        from data.mixer import MixConfig
        cfg = MixConfig.from_yaml(_MIX_CONFIGS / "cunca_pretraining.yaml")
        assert cfg.name == "cunca_pretraining"
        langs = [s.lang for s in cfg.sources]
        assert "gl" in langs
        assert "pt" in langs

    def test_cunca_gl_weight_dominant(self):
        from data.mixer import MixConfig
        cfg = MixConfig.from_yaml(_MIX_CONFIGS / "cunca_pretraining.yaml")
        nw = cfg.weights_normalized
        gl_idx = next(i for i, s in enumerate(cfg.sources) if s.lang == "gl")
        assert nw[gl_idx] >= 0.55, "Galician should be ≥55% in CUNCA pretraining"

    def test_cunca_weights_sum_to_one(self):
        from data.mixer import MixConfig
        cfg = MixConfig.from_yaml(_MIX_CONFIGS / "cunca_pretraining.yaml")
        nw = cfg.weights_normalized
        assert abs(sum(nw) - 1.0) < 1e-6
