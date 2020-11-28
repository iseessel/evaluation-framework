import os
# Upload containers to docker.
for chunk in range(0,10):
  os.system(f"docker build --build-arg CHUNK_NUMBER={chunk} -t fbprophet-{chunk} .")
  os.system(f"docker tag fbprophet-{chunk} gcr.io/silicon-badge-274423/fbprophet-{chunk}")
  os.system(f"docker push gcr.io/silicon-badge-274423/fbprophet-{chunk}")
