# Kubernetes deployment notes

These manifests deploy the API and worker containers. Provision Redis and PostgreSQL as managed services or apply organization-approved stateful workloads separately. Create the `eta-platform` namespace and a `delivery-eta-secrets` Secret containing `DATABASE_URL` and `REDIS_URL`; do not commit real credentials. Replace the example image with the published image tag before applying. These manifests are a portfolio starting point, not a production security review.
