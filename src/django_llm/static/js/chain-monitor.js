class ChainMonitor {
    constructor(chainId) {
        this.chainId = chainId;
        this.callbacks = {
            onStart: () => {},
            onStep: () => {},
            onComplete: () => {},
            onError: () => {}
        };
        
        this.connect();
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.socket = new WebSocket(
            `${protocol}//${window.location.host}/ws/chain/${this.chainId}/`
        );

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            switch(data.update_type) {
                case 'start':
                    this.callbacks.onStart(data);
                    break;
                case 'step':
                    this.callbacks.onStep(data);
                    break;
                case 'complete':
                    this.callbacks.onComplete(data);
                    break;
                case 'error':
                    this.callbacks.onError(data);
                    break;
            }
        };
    }

    onStart(callback) {
        this.callbacks.onStart = callback;
    }

    onStep(callback) {
        this.callbacks.onStep = callback;
    }

    onComplete(callback) {
        this.callbacks.onComplete = callback;
    }

    onError(callback) {
        this.callbacks.onError = callback;
    }
} 