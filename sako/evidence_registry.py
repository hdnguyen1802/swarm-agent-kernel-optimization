"""Use one typed set of evidence IDs for prompt references and validation."""


class EvidenceRegistry:
    KINDS = {"hardware", "contract", "baseline", "candidate", "artifact", "lesson", "diagnostic"}

    def __init__(self):
        self._kinds = {}

    def add(self, identifier, kind):
        if kind not in self.KINDS or not isinstance(identifier, str) or not identifier:
            raise ValueError("Evidence needs a nonempty ID and a supported type.")
        self._kinds.setdefault(identifier, set()).add(kind)
        return identifier

    def ids(self):
        return set(self._kinds)

    def types(self):
        return {kind: sorted(key for key, kinds in self._kinds.items() if kind in kinds)
                for kind in sorted(self.KINDS) if any(kind in kinds for kinds in self._kinds.values())}


def coordinator_evidence(task_id, hardware, candidates, lessons, artifacts, diagnostic=None):
    registry = EvidenceRegistry()
    registry.add("hardware:" + hardware.get("fingerprint", "unknown"), "hardware")
    registry.add("contract:" + task_id, "contract")
    registry.add("baseline:" + task_id, "baseline")
    for candidate in candidates:
        registry.add(candidate["id"], "candidate")
    for lesson in lessons:
        identifier = lesson.get("id", lesson.get("lesson_id"))
        if identifier:
            registry.add(identifier, "lesson")
    for artifact in artifacts:
        registry.add(artifact["path"], "artifact")
    if diagnostic:
        registry.add("diagnostic:" + task_id, "diagnostic")
    return registry
