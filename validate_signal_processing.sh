#!/bin/bash
set -o pipefail
cd "$(dirname "$0")" || exit 1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lilia-matplotlib}"

echo "====== VALIDATION: Signal Processing Consolidation ======"
echo ""

# 1. py_compile validation
echo "1) PY_COMPILE VALIDATION"
echo "========================"
py_compile_passed=0
py_compile_failed=0

for file in lilia/signal.py extract_tflite_signal_pipeline.py plot_tflite_summary.py data_analysis.py; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo "✓ $file"
        ((py_compile_passed++))
    else
        echo "✗ $file"
        ((py_compile_failed++))
    fi
done

echo "  Result: $py_compile_passed passed, $py_compile_failed failed"
echo ""

# 2. help smoke test
echo "2) HELP SMOKE TEST"
echo "=================="
help_passed=0
help_failed=0

for script in extract_tflite_signal_pipeline.py plot_tflite_summary.py data_analysis.py; do
    if python "$script" --help >/dev/null 2>&1; then
        echo "✓ $script --help"
        ((help_passed++))
    else
        echo "✗ $script --help"
        ((help_failed++))
    fi
done

echo "  Result: $help_passed passed, $help_failed failed"
echo ""

# 3. Unit tests
echo "3) UNIT TESTS"
echo "============="
python -m unittest discover -s tests -p "test_*.py" -v 2>&1 | tee "${LILIA_TEST_LOG:-/tmp/lilia-test-output.txt}"
test_exit_code=$?

# Count test results
test_passed=$(grep -c "^test.*\.\.\. ok$" "${LILIA_TEST_LOG:-/tmp/lilia-test-output.txt}" 2>/dev/null || true)

echo ""
echo "====== SUMMARY ======"
echo "py_compile: $py_compile_passed passed, $py_compile_failed failed"
echo "help smoke: $help_passed passed, $help_failed failed"
echo "unit tests: ~$test_passed passed"
echo "test exit code: $test_exit_code"

if (( py_compile_failed > 0 || help_failed > 0 || test_exit_code != 0 )); then
    exit 1
fi
exit 0
