import os
import sys
import socket
import argparse
import http.server
import socketserver
import threading
import time
import signal
from urllib.parse import quote

class AudioServerHandler(http.server.SimpleHTTPRequestHandler):
    """A custom HTTP request handler that serves audio files."""
    
    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, directory=directory, **kwargs)
    
    def log_message(self, format, *args):
        # Suppress log messages to keep console output clean
        pass
    
    def do_GET(self):
        # Serve audio files from the directory
        super().do_GET()

class AudioServer:
    """A simple HTTP server for serving audio files temporarily."""
    
    def __init__(self, host='0.0.0.0', port=8000, directory='.', timeout=3600):
        self.host = host
        self.port = port
        self.directory = os.path.abspath(directory)
        self.timeout = timeout
        self.server = None
        self.server_thread = None
        self.stop_event = threading.Event()
    
    def get_file_url(self, filepath):
        """
        Convert a filepath to a URL accessible through this server.
        
        Args:
            filepath: Absolute path to the file
            
        Returns:
            URL string that can be used to access the file
        """
        if not os.path.isabs(filepath):
            filepath = os.path.abspath(filepath)
        
        # Check that the file exists and is within the server directory
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not filepath.startswith(self.directory):
            raise ValueError(f"File {filepath} is outside the server directory {self.directory}")
        
        # Get the file path relative to the server directory
        relpath = os.path.relpath(filepath, self.directory)
        
        # URL encode the path
        url_path = quote(relpath)
        
        # Get the server's external IP
        external_ip = self.get_external_ip()
        
        return f"http://{external_ip}:{self.port}/{url_path}"
    
    def get_external_ip(self):
        """
        Get the machine's external IP address.
        In cloud environments, this will try to use the public IP from environment variable.
        """
        # First check if PUBLIC_IP environment variable is set
        public_ip = os.environ.get('PUBLIC_IP')
        if public_ip:
            return public_ip
            
        # Otherwise try to get the local network IP
        if self.host == '0.0.0.0':
            try:
                # Try to get the actual machine IP for better external access
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # Doesn't need to be reachable, just to determine the interface
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
                s.close()
                print(f"WARNING: Using local IP address {ip}. This will not work if Replicate cannot reach your network.")
                print("Set the PUBLIC_IP environment variable to your server's public IP address.")
                return ip
            except:
                print("WARNING: Using localhost. This will not work with Replicate!")
                print("Set the PUBLIC_IP environment variable to your server's public IP address.")
                return 'localhost'
        return self.host
    
    def start(self):
        """Start the audio server in a separate thread."""
        if self.server is not None:
            print("Server is already running")
            return
        
        # Create and configure the server
        handler = lambda *args, **kwargs: AudioServerHandler(*args, directory=self.directory, **kwargs)
        
        # Try to create the server, handle case where port is already in use
        for attempt in range(10):  # Try up to 10 different ports
            try:
                self.server = socketserver.TCPServer((self.host, self.port), handler)
                break
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    print(f"Port {self.port} is already in use, trying next port...")
                    self.port += 1
                else:
                    raise
        
        if self.server is None:
            raise RuntimeError("Failed to start the server after multiple attempts")
        
        # Set up auto-shutdown timer
        def shutdown_timer():
            start_time = time.time()
            while not self.stop_event.is_set():
                if time.time() - start_time > self.timeout:
                    print(f"Server timeout reached ({self.timeout}s), shutting down")
                    self.stop()
                    break
                time.sleep(1)
        
        # Start server in a separate thread
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Start timeout thread
        self.timeout_thread = threading.Thread(target=shutdown_timer)
        self.timeout_thread.daemon = True
        self.timeout_thread.start()
        
        print(f"Audio server started at http://{self.get_external_ip()}:{self.port}/")
        print(f"Serving files from {self.directory}")
        print(f"Server will auto-shutdown after {self.timeout} seconds of inactivity")
    
    def stop(self):
        """Stop the audio server."""
        if self.server is None:
            print("Server is not running")
            return
        
        self.stop_event.set()
        self.server.shutdown()
        self.server.server_close()
        self.server = None
        self.server_thread = None
        print("Audio server stopped")

# Global server instance
audio_server = None

def start_server(directory='.', port=8000, timeout=3600):
    """Start the audio server with the given parameters."""
    global audio_server
    if audio_server is not None:
        print("Server is already running, stopping first")
        audio_server.stop()
    
    audio_server = AudioServer(port=port, directory=directory, timeout=timeout)
    audio_server.start()
    return audio_server

def stop_server():
    """Stop the currently running server."""
    global audio_server
    if audio_server is not None:
        audio_server.stop()
        audio_server = None

def get_url_for_file(filepath):
    """Get a URL for accessing the given file through the server."""
    global audio_server
    if audio_server is None:
        raise RuntimeError("Server is not running")
    return audio_server.get_file_url(filepath)

def handle_signal(sig, frame):
    """Handle termination signals to clean up the server."""
    print("Received signal to terminate, stopping server")
    stop_server()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    parser = argparse.ArgumentParser(description="Start a simple HTTP server for serving audio files")
    parser.add_argument("--directory", type=str, default=".", help="Directory to serve files from")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--timeout", type=int, default=3600, help="Auto-shutdown timeout in seconds")
    args = parser.parse_args()
    
    print(f"Starting audio server from directory: {args.directory}")
    server = start_server(directory=args.directory, port=args.port, timeout=args.timeout)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping server")
        stop_server()