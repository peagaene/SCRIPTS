import tempfile
from pathlib import Path


class TestRegressionSuite:
    def test_txt_parser_backward_compatibility(self):
        sample_txt_data = """
TYPE;E;N;Z
1 PV 123;345678.12;7654321.98;850.5
TV;345680.00;7654320.00;850.0
ARVORE;345690.00;7654330.00;851.2
"""
        from reurb.io.txt_parser import ler_txt

        with tempfile.TemporaryDirectory() as td:
            temp_file = Path(td) / "test.txt"
            temp_file.write_text(sample_txt_data, encoding="utf-8")
            df = ler_txt(str(temp_file))

        assert len(df) == 3
        assert "TYPE" in df.columns
        assert "E" in df.columns
        assert df.iloc[0]["TYPE"] == "1 PV 123"

    def test_layer_mapping_preserves_old_keys(self):
        from reurb.config.mappings import TYPE_TO_LAYER

        old_keys = {
            "PA",
            "PI",
            "PFI",
            "PVE",
            "PVAP",
            "AEPVE",
            "AEBO",
            "AEBO1",
            "BL1",
        }

        for key in old_keys:
            assert key in TYPE_TO_LAYER, f"Chave antiga '{key}' foi removida!"

    def test_geometry_calculations_preserve_precision(self):
        from reurb.geometry.calculations import clamp, calcular_offset

        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

        p1, p2 = (0, 0), (10, 0)
        ox, oy = calcular_offset(p1, p2, dist=1.0)
        assert abs(oy - 1.0) < 1e-6
        assert abs(ox) < 1e-6
