# AppArmor profile for OpenHands agent-server terminal sandbox.
#
# Default-allow with an explicit denylist: the agent can read/write its
# configurable work_dir and standard system tools, but is blocked from
# high-value host secrets and privilege-escalation utilities. This is
# intentionally a denylist (not an allowlist) so arbitrary work_dir values
# work without per-conversation profile reloading — reloading would require
# CAP_MAC_ADMIN at runtime, which restricted pods typically do not grant.
#
# Loaded into the kernel at image build time by `apparmor_parser -r -W`.
# Used at runtime via `aa-exec -p openhands-agent-terminal -- bash -i`.
#
# Flags:
#   attach_disconnected — handle paths seen through bind mounts or chroots.
#   mediate_deleted       — continue enforcing even if the profiled binary is
#                           replaced under a running process.
profile openhands-agent-terminal flags=(attach_disconnected,mediate_deleted) {

  # ------------------------------------------------------------------
  # High-value host secrets (deny even if the work_dir is misconfigured
  # to / or overlaps with sensitive paths).
  # ------------------------------------------------------------------
  deny /etc/shadow                  rw,
  deny /etc/shadow-                 rw,
  deny /etc/gshadow                 rw,
  deny /etc/gshadow-                rw,
  deny /etc/sudoers                 rw,
  deny /etc/sudoers.d/**            rw,
  deny /etc/master.passwd           rw,
  deny /etc/security/**             rw,

  # ------------------------------------------------------------------
  # Per-user secret directories (SSH, cloud CLIs, k8s, GPG).
  # ------------------------------------------------------------------
  deny /root/**                                     rw,
  deny /home/*/.ssh/**                              rw,
  deny /home/*/.aws/**                              rw,
  deny /home/*/.gnupg/**                            rw,
  deny /home/*/.kube/**                             rw,
  deny /home/*/.config/gcloud/**                    rw,
  deny /home/*/.docker/config.json                  rw,
  deny /home/*/.npmrc                               rw,
  deny /home/*/.pypirc                              rw,
  deny /home/*/.netrc                               rw,

  # ------------------------------------------------------------------
  # Secrets-bearing files that can appear anywhere (env files, cred dumps).
  # ------------------------------------------------------------------
  deny /**/.env             rw,
  deny /**/.env.*           rw,
  deny /**/credentials.json rw,
  deny /**/service-account*.json rw,

  # ------------------------------------------------------------------
  # Block privilege-escalation utilities. `x` denial prevents execve();
  # the agent shell can still use builtins and ordinary binaries.
  # ------------------------------------------------------------------
  deny /usr/bin/sudo        x,
  deny /usr/bin/su          x,
  deny /usr/bin/passwd      x,
  deny /usr/bin/chroot      x,
  deny /usr/bin/nsenter     x,
  deny /usr/bin/unshare     x,
  deny /usr/bin/newuidmap   x,
  deny /usr/bin/newgidmap   x,
  deny /usr/sbin/useradd    x,
  deny /usr/sbin/usermod    x,
  deny /usr/sbin/userdel    x,
  deny /usr/sbin/groupadd   x,
  deny /usr/sbin/groupmod   x,
  deny /usr/sbin/visudo     x,

  # ------------------------------------------------------------------
  # Everything else: allow read/execute anywhere, write anywhere.
  # The agent's work_dir (wherever it is) is writable via this fallback.
  # ------------------------------------------------------------------
  /** rix,

  # Network is allowed — agents routinely fetch packages / call APIs.
  # To build a strict offline variant, copy this profile and replace the
  # line below with `deny network,`.
  network,

  # Signals: allow the agent to signal its own descendants (used by
  # terminal interrupt / Ctrl+C flow).
  signal (send, receive) peer=openhands-agent-terminal,

  # ptrace: allow self-trace only (some debuggers / `strace` need it);
  # deny tracing other profiles.
  ptrace (readby, tracedby) peer=openhands-agent-terminal,
}
