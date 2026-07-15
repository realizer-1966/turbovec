# SIMD 커널 아키텍처

turbovec의 SIMD 검색 커널은 아키텍처별로 최적화되어 있습니다.

## ARM (NEON)

ARM에서는 **NEON 명령어**를 사용하여 순차적 코드 레이아웃으로 검색합니다. 이는 ARM의 SIMD 설계에 가장 잘 맞는 방식입니다.

> NEON 커널은 FAISS IndexPQFastScan보다 10-19% 빠릅니다.

## x86 (AVX-512BW / AVX2)

x86에서는 두 가지 경로가 있습니다:

1. **AVX-512BW** — 사용 가능한 경우 우선 선택됩니다.
2. **AVX2 폴백** — FAISS 스타일 perm0-interleaved 레이아웃을 사용합니다.

```rust
// 런타임 감지로 커널 선택
if is_x86_feature_detected!("avx512bw") {
    // AVX-512BW 경로
} else if is_x86_feature_detected!("avx2") {
    // AVX2 폴백
} else {
    // 스칼라 폴백
}
```

### 스칼라 폴백

SIMD가 없는 플랫폼에서는 스칼라 구현이 사용됩니다. x86에서는 `#[cfg(test)]`로 강제 활성화할 수 있어 테스트가 가능합니다.

## 마스크 기반 필터링

검색 시 `allowlist`나 `slot bitmask`를 전달하면:

- 32-벡터 블록 단위로 early-exit
- 허용된 슬롯이 없는 블록은 건너뜀
- `BLOCKS_SKIPPED_BY_MASK` 카운터로 스킵 통계 제공

![diagram](https://example.com/diagram.png)

자세한 내용은 [SIMD 문서](https://example.com)를 참조하세요.

---

*성능 벤치마크는 ARM Cortex-A78에서 측정되었습니다.*