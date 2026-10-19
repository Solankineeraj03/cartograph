# Failed run: Ray Serve start (2)

Ray 2.49 Serve config code calls FieldDescriptor.label, which protobuf 7.36 (pulled in unpinned via Feast) removed.
Fix: pin protobuf 5.29.5 in envs/serve_app.in.
