from .core import LogAnalyzer


def run_analysis() -> None:
    analyzer = LogAnalyzer(padding=50)

    df = analyzer.load_and_merge_logs()

    if df is not None:
        roi = analyzer.calculate_roi(df)
        if roi is not None:
            analyzer.output_result(roi)
            analyzer.save_roi(roi)
