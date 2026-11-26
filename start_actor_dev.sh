#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
: "${PYROMIND_SANDBOX_ACTOR_URL:?请先 export 远端沙箱路由地址}"
: "${PYROMIND_SANDBOX_ACTOR_TARGET:?请先 export 个人 atespace/actor}"
export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"
if [[ ! -f start_inference.sh ]]; then
  echo "请把 start_actor_dev.sh 与个人的 start_inference.sh 放在同一目录。" >&2
  exit 1
fi

actor_url="${PYROMIND_SANDBOX_ACTOR_URL%/}"
forward_pid=""
kubeconfig_temp=""
cleanup() {
  if [[ -n "${forward_pid}" ]]; then
    kill "${forward_pid}" 2>/dev/null || true
    wait "${forward_pid}" 2>/dev/null || true
  fi
  if [[ -n "${kubeconfig_temp}" ]]; then
    rm -f -- "${kubeconfig_temp}"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${KUBECONFIG}" == *$'\n'* ]]; then
  mkdir -p .agent_tmp
  kubeconfig_temp="$(umask 077; mktemp "${PWD}/.agent_tmp/actor-kubeconfig.XXXXXX")"
  printf '%s' "${KUBECONFIG}" > "${kubeconfig_temp}"
  export KUBECONFIG="${kubeconfig_temp}"
fi

if [[ "${actor_url}" == "http://127.0.0.1:8081" ]] &&
   ! curl --noproxy 127.0.0.1,localhost -s --connect-timeout 1 --max-time 2 \
     -o /dev/null "${actor_url}/healthz"; then
  mkdir -p .agent_tmp
  kubectl -n ate-system port-forward --address 127.0.0.1 svc/atenet-router 8081:80 \
    > .agent_tmp/actor-dev-forward.log 2>&1 &
  forward_pid=$!
fi

ready=0
for attempt in {1..30}; do
  if curl --noproxy 127.0.0.1,localhost -fsS --connect-timeout 1 --max-time 2 -o /dev/null \
    -H "ate-target-actor: ${PYROMIND_SANDBOX_ACTOR_TARGET}" "${actor_url}/readyz" 2>/dev/null; then
    ready=1
    break
  fi
  if [[ -n "${forward_pid}" ]] && ! kill -0 "${forward_pid}" 2>/dev/null; then
    echo "转发失败，请查看 .agent_tmp/actor-dev-forward.log。" >&2
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "沙箱 ${PYROMIND_SANDBOX_ACTOR_TARGET} 未就绪，请检查路由、Actor 和 KubeConfig。" >&2
  exit 1
fi

echo "远端沙箱 ${PYROMIND_SANDBOX_ACTOR_TARGET} 已就绪，启动 start_inference.sh。"
PYROMIND_PI_TERMINAL_BACKEND=sandbox bash ./start_inference.sh
