from openpyxl import Workbook

from survey_data import REQUIRED_COLUMNS, load_survey_rows


def test_loader_reads_validated_deidentified_workbook(tmp_path):
    path = tmp_path / "survey.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sessions + Survey 1 + Survey 2"
    headers = sorted(REQUIRED_COLUMNS | {"username"})
    sheet.append(headers)
    row = {header: 3 for header in headers}
    row["session_id"] = "session-a"
    row["S1_pre_calm_tense_conversation"] = 75
    row["S2_post_calm_tense_conversation"] = 80
    sheet.append([row[header] for header in headers])
    workbook.save(path)

    rows, diagnostics = load_survey_rows(path)

    assert rows[0]["session_id"] == "session-a"
    assert "username" not in rows[0]
    assert diagnostics["valid_session_ids"] == 1
