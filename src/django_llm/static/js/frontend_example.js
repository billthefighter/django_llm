// Example implementation of chain monitoring with UI updates
document.addEventListener('DOMContentLoaded', () => {
    // Initialize chain monitor for a specific chain execution
    const chainId = document.getElementById('chain-container').dataset.chainId;
    const monitor = new ChainMonitor(chainId);

    // UI Elements
    const statusEl = document.getElementById('chain-status');
    const progressEl = document.getElementById('chain-progress');
    const resultEl = document.getElementById('chain-result');
    const errorEl = document.getElementById('chain-error');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Handle chain start
    monitor.onStart((data) => {
        statusEl.textContent = 'Chain Execution Started';
        loadingIndicator.style.display = 'block';
        errorEl.style.display = 'none';
        
        // Display any initial metadata
        if (data.metadata) {
            console.log('Chain metadata:', data.metadata);
        }
    });

    // Handle step updates
    monitor.onStep((data) => {
        // Update progress information
        progressEl.textContent = `Executing step: ${data.step_name || 'Unknown'}`;
        
        // If you have a progress bar
        if (data.progress_percentage) {
            progressEl.style.width = `${data.progress_percentage}%`;
        }

        // Display intermediate results if available
        if (data.intermediate_result) {
            resultEl.innerHTML += `
                <div class="step-result">
                    <strong>${data.step_name}:</strong>
                    <pre>${JSON.stringify(data.intermediate_result, null, 2)}</pre>
                </div>
            `;
        }
    });

    // Handle completion
    monitor.onComplete((data) => {
        statusEl.textContent = 'Chain Execution Completed';
        loadingIndicator.style.display = 'none';

        // Display final result
        resultEl.innerHTML += `
            <div class="final-result">
                <h3>Final Result:</h3>
                <pre>${JSON.stringify(data.result, null, 2)}</pre>
            </div>
        `;
    });

    // Handle errors
    monitor.onError((data) => {
        statusEl.textContent = 'Chain Execution Failed';
        loadingIndicator.style.display = 'none';
        
        // Display error message
        errorEl.style.display = 'block';
        errorEl.innerHTML = `
            <div class="alert alert-danger">
                Error: ${data.error}
            </div>
        `;
    });
}); 