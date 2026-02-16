#!/bin/bash
# Training progress monitor

OUTPUT_FILE="/private/tmp/claude-501/-Users-antipop-src-github-com-kentaro-microgpt-ex/tasks/b3b5d50.output"

while true; do
    if [ -f "$OUTPUT_FILE" ]; then
        clear
        echo "=== MicroGPT Training Progress ==="
        echo ""

        # Get current step
        CURRENT_STEP=$(grep -oE "step\s+[0-9]+" "$OUTPUT_FILE" | tail -1 | awk '{print $2}')

        if [ -n "$CURRENT_STEP" ]; then
            PROGRESS=$(echo "scale=1; $CURRENT_STEP * 100 / 1000" | bc)

            echo "Current Step: $CURRENT_STEP / 1000 ($PROGRESS%)"
            echo ""

            # Show last 10 steps
            echo "Last 10 steps:"
            tail -10 "$OUTPUT_FILE" | grep "step"
            echo ""

            # Calculate average loss from last 10 steps
            AVG_LOSS=$(tail -10 "$OUTPUT_FILE" | grep "loss" | awk '{print $NF}' | awk '{sum+=$1} END {if(NR>0) printf "%.4f", sum/NR}')
            echo "Average loss (last 10): $AVG_LOSS"

            # Estimate time remaining
            ELAPSED_SECONDS=$((CURRENT_STEP * 5))
            REMAINING_STEPS=$((1000 - CURRENT_STEP))
            REMAINING_SECONDS=$((REMAINING_STEPS * 5))
            REMAINING_MINUTES=$((REMAINING_SECONDS / 60))

            echo "Estimated time remaining: ~${REMAINING_MINUTES} minutes"
        else
            echo "Waiting for training to start..."
        fi
    else
        echo "Output file not found: $OUTPUT_FILE"
    fi

    sleep 30
done
