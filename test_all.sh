#!/bin/bash
# Test script for both assignments

echo "=================================="
echo "Testing Medical XAI Assignment 1"
echo "=================================="

cd assignment1_xai
python xai_chest_xray.py

if [ $? -eq 0 ]; then
    echo "✓ Assignment 1 completed successfully!"
    echo "  Check outputs/xai_visualizations/ for results"
else
    echo "✗ Assignment 1 failed"
    exit 1
fi

cd ..

echo ""
echo "=================================="
echo "Testing Pruning Assignment 2"
echo "=================================="

cd assignment2_pruning
python pruning_demo.py

if [ $? -eq 0 ]; then
    echo "✓ Assignment 2 completed successfully!"
    echo "  Check outputs/pruning_comparison/ for results"
else
    echo "✗ Assignment 2 failed"
    exit 1
fi

cd ..

echo ""
echo "=================================="
echo "All tests completed successfully!"
echo "=================================="
echo ""
echo "Generated files:"
echo "- outputs/xai_visualizations/xai_comparison.png"
echo "- outputs/pruning_comparison/pruning_comparison.png"
echo ""
echo "To view results:"
echo "  ls -la outputs/xai_visualizations/"
echo "  ls -la outputs/pruning_comparison/"
