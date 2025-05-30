#!/bin/bash

# Performance testing script for MV-Adapter - dual run comparison
echo "MV-Adapter Performance Testing - Dual Run"

# Function to get memory usage by process
get_memory_usage() {
    local pid=$1
    if [ -n "$pid" ] && ps -p $pid > /dev/null 2>&1; then
        ps -p $pid -o rss= | awk '{print $1/1024}' 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Function to get VRAM usage
get_vram_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}'
    else
        echo "0"
    fi
}

# Function to run single test
run_test() {
    local test_name=$1
    local use_mmgp=$2
    local output_suffix=$3
    
    echo "========================================="
    echo "Running $test_name"
    echo "========================================="
    
    # Memory monitoring setup
    log_file="memory_${output_suffix}.log"
    echo "timestamp,ram_mb,vram_mb" > $log_file
    
    # Build command
    cmd="python -m scripts.inference_ig2mv_sdxl \
        --image assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.jpeg \
        --mesh assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.glb \
        --output output_${output_suffix}.png \
        --remove_bg"
    
    if [ "$use_mmgp" = "true" ]; then
        cmd="$cmd --mmgp"
    fi
    
    # Start test
    start_time=$(date +%s.%N)
    $cmd &
    python_pid=$!
    
    # Monitor memory
    max_ram=0
    max_vram=0
    
    while ps -p $python_pid > /dev/null 2>&1; do
        current_time=$(date +%s.%N)
        current_ram=$(get_memory_usage $python_pid)
        current_vram=$(get_vram_usage)
        
        echo "${current_time},${current_ram},${current_vram}" >> $log_file
        
        if (( $(echo "$current_ram > $max_ram" | bc -l) )); then
            max_ram=$current_ram
        fi
        
        if (( $current_vram > $max_vram )); then
            max_vram=$current_vram
        fi
        
        sleep 1
    done
    
    # Wait for completion
    wait $python_pid
    exit_code=$?
    
    # Calculate execution time
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc -l)
    
    # Display results
    echo "Results for $test_name:"
    echo "  Execution time: ${execution_time} seconds"
    echo "  Max RAM usage: ${max_ram} MB"
    echo "  Max VRAM usage: ${max_vram} MB"
    echo "  Exit code: ${exit_code}"
    echo ""
    
    # Save results
    {
        echo "Test: $test_name"
        echo "Date: $(date)"
        echo "Execution time: ${execution_time} seconds"
        echo "Max RAM usage: ${max_ram} MB"
        echo "Max VRAM usage: ${max_vram} MB"
        echo "Exit code: ${exit_code}"
        echo "Log file: ${log_file}"
        echo "========================================="
    } >> "dual_test_results.txt"
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $test_name completed successfully"
        if [ -f "output_${output_suffix}.png" ]; then
            ls -lh "output_${output_suffix}.png"
        fi
    else
        echo "✗ $test_name failed with exit code: ${exit_code}"
    fi
}

# Initialize results file
echo "Dual Performance Test Results" > "dual_test_results.txt"
echo "Generated: $(date)" >> "dual_test_results.txt"
echo "=========================================" >> "dual_test_results.txt"

# Run tests
echo "Starting dual performance tests..."
echo ""

# Test 1: Without mmgp
# run_test "Test without mmgp" "false" "no_mmgp"

# Small delay between tests
# sleep 5

# Test 2: With mmgp
run_test "Test with mmgp" "true" "with_mmgp"

echo "========================================="
echo "DUAL TEST COMPLETE"
echo "========================================="
echo "Results saved to: dual_test_results.txt"
echo "Memory logs saved to: memory_no_mmgp.log and memory_with_mmgp.log"
echo "Output images: output_no_mmgp.png and output_with_mmgp.png" 