# Failed run: Ray Serve start

The serve_app env resolved FastAPI 0.143.3. @serve.ingress(FastAPI()) failed: Ray 2.49 could not pickle the FastAPI app
(_native_telemetry holds a _thread.lock). Fix: route in the deployment __call__ with Starlette request/response types.
Redis materialization in this run succeeded.
