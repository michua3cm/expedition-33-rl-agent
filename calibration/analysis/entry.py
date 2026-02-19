from .core import LogAnalyzer

def run_analysis():
    # Instantiate the analyzer
    analyzer = LogAnalyzer(padding=50)

    # 1. Load
    df = analyzer.load_and_merge_logs()

    # 2. Analyze
    if df is not None:
        roi = analyzer.calculate_roi(df)

        # 3. Output
        analyzer.output_result(roi)