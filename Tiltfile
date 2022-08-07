context = k8s_context()
allow_k8s_contexts(context)

k8s_yaml(helm('./helm'))

# gcr / kaniko image build in GCP
gcloud_build = (
    """
    gcloud builds submit \
    --config cloudbuild.yaml \
    --substitutions=TAG_NAME=$EXPECTED_TAG
    """
)

# live_update in dev environment
custom_build(
    ref='gcr.io/windmark/torch',
    command=gcloud_build,
    skips_local_docker=True,
    disable_push=True,
    deps=[
        './requirements.txt',
        './helm/',
        './Dockerfile',
        './Tiltfile',
        './src/',
    ],
    ignore=['./docs/'],
    live_update=[
        sync('./src/', '/app/'),
        run('date > /trigger.txt'),
    ],
)
