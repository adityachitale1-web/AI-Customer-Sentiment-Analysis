#!/bin/bash
# Double-click to stop the CXSentinel servers.
lsof -ti :8000 | xargs kill 2>/dev/null && echo "• API stopped" || echo "• API was not running"
lsof -ti :8501 | xargs kill 2>/dev/null && echo "• Dashboard stopped" || echo "• Dashboard was not running"
echo "Done."
