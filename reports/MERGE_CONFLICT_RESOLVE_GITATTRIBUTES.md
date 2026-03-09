# .gitattributes Merge Conflict 해결 방법

Hugging Face PR에서 **This branch has merge conflicts in .gitattributes** 가 나올 때 아래 순서대로 하면 됩니다.

---

## 1. PR 페이지에서 Conflict 해결

1. https://huggingface.co/pathcosmos/frankenstallm **Pull requests** 로 이동
2. Conflict 나는 PR 클릭
3. **Resolve conflicts** 또는 **Conflict 해결** 버튼 클릭
4. `.gitattributes` 파일이 열리면, **전체 내용을 지우고** 아래 블록 **전체**로 교체

---

## 2. 사용할 .gitattributes 내용 (전체 복사)

```
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.gguf filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.7z filter=lfs diff=lfs merge=lfs -text
```

5. **Mark as resolved** / **충돌 해결됨** 체크 후 저장
6. **Merge pull request** 로 머지 진행

---

## 요약

- **원인**: main과 PR 브랜치에 서로 다른 `.gitattributes` 가 있어서 충돌 발생.
- **해결**: 위 내용으로 **통일**하면 LFS 규칙만 남고 충돌은 사라짐.
- `*.safetensors`, `*.gguf` 등 우리가 올린 대용량 파일이 LFS로 잘 올라가도록 위 설정이면 충분합니다.
