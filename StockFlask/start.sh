#!/bin/bash
#!/bin/bash
# Use waitress to serve the app on the Render-provided port
waitress-serve --listen=0.0.0.0:$PORT app:app

