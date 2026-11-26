from collections.abc import Callable
from datetime import UTC, datetime

from pyromind_runtime.domain.project_storage import (
    ProjectStorageBinding,
    StorageProxyConfig,
)
from pyromind_runtime.ports.project_storage import (
    ProjectStorageBindingStore,
    StorageBindingBlocked,
    StorageBindingVerifier,
)


class ProjectStorageBindings:
    def __init__(
        self,
        store: ProjectStorageBindingStore,
        verifier_factory: Callable[[str, str], StorageBindingVerifier],
    ) -> None:
        self.store = store
        self.verifier_factory = verifier_factory

    def get(self, project_id: str, owner_user_id: str) -> ProjectStorageBinding | None:
        binding = self.store.load(project_id)
        if binding is not None and binding.owner_user_id != owner_user_id:
            raise PermissionError("Storage binding belongs to another user")
        return binding

    def bind(
        self, project_id: str, owner_user_id: str, config: StorageProxyConfig
    ) -> ProjectStorageBinding:
        with self.store.lease(project_id):
            self.get(project_id, owner_user_id)
            verifier = self.verifier_factory(project_id, owner_user_id)
            binding = ProjectStorageBinding(
                project_id=project_id,
                owner_user_id=owner_user_id,
                cluster=verifier.cluster,
                config=config,
                last_checked_at=datetime.now(UTC),
            )
            return self._verify(binding, verifier)

    def refresh(self, project_id: str, owner_user_id: str) -> ProjectStorageBinding:
        with self.store.lease(project_id):
            binding = self.get(project_id, owner_user_id)
            if binding is None:
                raise ValueError("Bind an existing Storage Sandbox first")
            return self._verify(
                binding, self.verifier_factory(project_id, owner_user_id)
            )

    def _verify(
        self, binding: ProjectStorageBinding, verifier: StorageBindingVerifier
    ) -> ProjectStorageBinding:
        error = None
        try:
            if binding.cluster != verifier.cluster:
                raise StorageBindingBlocked(
                    "Project cluster changed; explicitly bind a Sandbox "
                    "in this cluster."
                )
            verifier.verify(binding.config)
        except Exception as exc:
            error = (
                str(exc)
                if isinstance(exc, StorageBindingBlocked)
                else "Storage Sandbox verification failed; check owner access and TLS."
            )
        binding = binding.model_copy(
            update={
                "status": "blocked" if error else "ready",
                "last_error": error,
                "last_checked_at": datetime.now(UTC),
            }
        )
        self.store.save(binding)
        return binding
