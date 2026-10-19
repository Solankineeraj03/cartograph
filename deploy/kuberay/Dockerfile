# CPU image for the deterministic Cartograph request path on KubeRay. Base pinned to the tested Ray version.
FROM rayproject/ray:2.49.2-py311-cpu
RUN pip install --no-cache-dir pandas==2.2.3 pyarrow==19.0.1 numpy==2.2.6 scipy==1.15.2 fastapi==0.115.12
COPY cartograph /app/cartograph
COPY data/demo_bundle_v1 /app/demo_bundle
ENV PYTHONPATH=/app CARTOGRAPH_BUNDLE=/app/demo_bundle
WORKDIR /app
