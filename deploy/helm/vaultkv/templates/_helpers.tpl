{{- define "vaultkv.name" -}}{{ .Release.Name }}-vaultkv{{- end -}}
{{- define "vaultkv.labels" -}}
app.kubernetes.io/name: vaultkv
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
{{- define "vaultkv.secretName" -}}
{{- if .Values.api.existingSecret -}}{{ .Values.api.existingSecret }}{{- else -}}{{ include "vaultkv.name" . }}-token{{- end -}}
{{- end -}}

{{- define "vaultkv.ingestPod" }}
securityContext: {fsGroup: 10001}
restartPolicy: OnFailure
containers:
  - name: ingest
    image: "{{ .ctx.Values.image.repository }}:{{ .ctx.Values.image.tag }}"
    imagePullPolicy: {{ .ctx.Values.image.pullPolicy }}
    command: ["/bin/sh", "-c"]
    args:
      - |
        set -e
        {{- range .ctx.Values.ingest.batches }}
        vaultkv ingest --db /data/vaultkv.sqlite3 --tenant {{ $.ctx.Values.ingest.tenant }} --user {{ $.ctx.Values.ingest.user }} --path "{{ $.root }}/{{ . }}" --name-root "{{ $.root }}"
        {{- end }}
    securityContext: {runAsNonRoot: true, runAsUser: 10001, allowPrivilegeEscalation: false}
    volumeMounts:
      - {name: data, mountPath: /data}
      {{- if eq .ctx.Values.ingest.source "hostPath" }}
      - {name: source, mountPath: /source, readOnly: true}
      {{- end }}
volumes:
  - name: data
    persistentVolumeClaim: {claimName: {{ include "vaultkv.name" .ctx }}-data}
  {{- if eq .ctx.Values.ingest.source "hostPath" }}
  - name: source
    hostPath: {path: {{ required "ingest.hostPath is required when source=hostPath" .ctx.Values.ingest.hostPath }}, type: Directory}
  {{- end }}
{{- end }}
