; Native CUDA W4A4 integer MMA and output indexing.
; Implementation: kernels/w4_integer_mma.cuh with PLAIT_W4_CUDA_INTEGER_MMA.
; A query asks for a counterexample to one source-level obligation.
; The PTX m16n8k64, dp4a and shuffle instruction semantics are assumed.
; Dependent device buffers must be live and nonoverlapping. C buffers are
; supplied by the checked Rust API. FP32 FMA and FP16 rounding are exercised
; by the device regression and are outside these integer obligations.
(set-logic QF_BV)

(define-fun signed-nibble ((x (_ BitVec 4))) (_ BitVec 32)
  ((_ sign_extend 28) x))
(define-fun unsigned-nibble ((x (_ BitVec 4))) (_ BitVec 32)
  ((_ zero_extend 28) x))

(declare-const activation (_ BitVec 4))
(declare-const weight (_ BitVec 4))
(declare-const zero (_ BitVec 4))

(define-fun mma-weight () (_ BitVec 32) (signed-nibble (bvxor weight #x8)))
(define-fun corrected-weight () (_ BitVec 32)
  (bvsub (unsigned-nibble weight) (unsigned-nibble zero)))

(echo "center-unsigned-weight")
(push)
(assert (not (= mma-weight (bvsub (unsigned-nibble weight) #x00000008))))
(check-sat)
(pop)

; Distributivity lifts this single term to every valid K group. The separate
; accumulator obligation checks that modular arithmetic cannot hide overflow.
(echo "asymmetric-zero-correction")
(push)
(assert
  (not (=
    (bvadd
      (bvmul (signed-nibble activation) mma-weight)
      (bvmul (bvsub #x00000008 (unsigned-nibble zero)) (signed-nibble activation)))
    (bvmul (signed-nibble activation) corrected-weight))))
(check-sat)
(pop)

(declare-const packed (_ BitVec 32))
(define-fun signed-byte ((x (_ BitVec 8))) (_ BitVec 32) ((_ sign_extend 24) x))
; dp4a.s32.s32 with multiplier bytes all one.
(define-fun add-four-bytes ((x (_ BitVec 32))) (_ BitVec 32)
  (bvadd
    (signed-byte ((_ extract 7 0) x))
    (signed-byte ((_ extract 15 8) x))
    (signed-byte ((_ extract 23 16) x))
    (signed-byte ((_ extract 31 24) x))))
(define-fun biased () (_ BitVec 32) (bvxor packed #x88888888))
(define-fun signed-shuffle-input () (_ BitVec 32)
  (bvadd
    (add-four-bytes (bvand biased #x0f0f0f0f))
    (add-four-bytes (bvand (bvlshr biased #x00000004) #x0f0f0f0f))
    #xffffffc0))

(echo "packed-activation-sum")
(push)
(assert
  (not (=
    signed-shuffle-input
    (bvadd
      (signed-nibble ((_ extract 3 0) packed))
      (signed-nibble ((_ extract 7 4) packed))
      (signed-nibble ((_ extract 11 8) packed))
      (signed-nibble ((_ extract 15 12) packed))
      (signed-nibble ((_ extract 19 16) packed))
      (signed-nibble ((_ extract 23 20) packed))
      (signed-nibble ((_ extract 27 24) packed))
      (signed-nibble ((_ extract 31 28) packed))))))
(check-sat)
(pop)

(echo "masked-k-tail-contributes-zero")
(push)
(assert (= activation #x0))
(assert (not (= (bvmul (signed-nibble activation) mma-weight) #x00000000)))
(check-sat)
(pop)

(reset)
(set-logic QF_LIA)

(define-fun lane-group ((lane Int)) Int (div lane 4))
(define-fun lane-in-group ((lane Int)) Int (mod lane 4))
(define-fun index-in ((i Int) (end Int)) Bool (and (<= 0 i) (< i end)))
(define-fun shuffle-one ((lane Int)) Int
  (+ lane (ite (= (mod lane 2) 0) 1 (- 1))))
(define-fun shuffle-two ((lane Int)) Int
  (+ lane (ite (< (lane-in-group lane) 2) 2 (- 2))))

; Source fragment array x[0..3], then the eight nibbles in each register.
(define-fun source-a-row ((lane Int) (i Int)) Int
  (+ (lane-group lane) (* 8 (mod (div i 8) 2))))
(define-fun source-a-col ((lane Int) (i Int)) Int
  (+ (* 8 (lane-in-group lane)) (* 32 (div i 16)) (mod i 8)))
; PTX ISA 8.8, "Matrix Fragments for mma.m16n8k64", signed four-bit A.
(define-fun ptx-a-row ((lane Int) (i Int)) Int
  (ite (or (< i 8) (and (<= 16 i) (< i 24))) (lane-group lane) (+ (lane-group lane) 8)))
(define-fun ptx-a-col ((lane Int) (i Int)) Int
  (ite (< i 16)
    (+ (* 8 (lane-in-group lane)) (mod i 8))
    (+ (* 8 (lane-in-group lane)) (mod i 8) 32)))

; Source fragment array y[0..1] and PTX signed four-bit B.
(define-fun source-b-row ((lane Int) (i Int)) Int
  (+ (* 8 (lane-in-group lane)) (* 32 (div i 8)) (mod i 8)))
(define-fun source-b-col ((lane Int)) Int (lane-group lane))
(define-fun ptx-b-row ((lane Int) (i Int)) Int
  (ite (< i 8)
    (+ (* 8 (lane-in-group lane)) (mod i 8))
    (+ (* 8 (lane-in-group lane)) (mod i 8) 32)))

(define-fun source-d-row ((lane Int) (i Int)) Int
  (+ (lane-group lane) (* 8 (div i 2))))
(define-fun source-d-col ((lane Int) (i Int)) Int
  (+ (* 2 (lane-in-group lane)) (mod i 2)))
(define-fun ptx-d-row ((lane Int) (i Int)) Int
  (ite (< i 2) (lane-group lane) (+ (lane-group lane) 8)))
(define-fun ptx-d-col ((lane Int) (i Int)) Int
  (+ (* 2 (lane-in-group lane)) (ite (or (= i 0) (= i 2)) 0 1)))

(declare-const lane Int)
(declare-const lane-other Int)
(declare-const i Int)
(declare-const i-other Int)
(assert (index-in lane 32))

(echo "warp-uniform-tile-loop")
(push)
(declare-const block Int)
(declare-const warp Int)
(assert (and (<= 0 block) (index-in warp 4)))
(assert (index-in lane-other 32))
(assert (not (=
  (+ (* block 4) (div (+ (* warp 32) lane) 32))
  (+ (* block 4) (div (+ (* warp 32) lane-other) 32)))))
(check-sat)
(pop)

; The two butterfly steps add a lane's own sum and all three peers' sums.
(echo "four-lane-shuffle-coverage")
(push)
(assert (not (and
  (= (lane-group lane) (lane-group (shuffle-one lane)))
  (= (lane-group lane) (lane-group (shuffle-two lane)))
  (= (lane-group lane) (lane-group (shuffle-two (shuffle-one lane))))
  (distinct lane (shuffle-one lane) (shuffle-two lane) (shuffle-two (shuffle-one lane))))))
(check-sat)
(pop)

(echo "mma-a-fragment")
(push)
(assert (index-in i 32))
(assert (not (and
  (= (source-a-row lane i) (ptx-a-row lane i))
  (= (source-a-col lane i) (ptx-a-col lane i))
  (index-in (source-a-row lane i) 16)
  (index-in (source-a-col lane i) 64))))
(check-sat)
(pop)

(echo "mma-b-fragment")
(push)
(assert (index-in i 16))
(assert (not (and
  (= (source-b-row lane i) (ptx-b-row lane i))
  (index-in (source-b-row lane i) 64)
  (index-in (source-b-col lane) 8))))
(check-sat)
(pop)

(echo "mma-d-fragment")
(push)
(assert (index-in i 4))
(assert (not (and
  (= (source-d-row lane i) (ptx-d-row lane i))
  (= (source-d-col lane i) (ptx-d-col lane i))
  (index-in (source-d-row lane i) 16)
  (index-in (source-d-col lane i) 8))))
(check-sat)
(pop)

; With 32 lanes and four valid outputs per lane, injection into the tile
; proves each of its 128 coordinates has exactly one writer.
(echo "output-tile-single-writer")
(push)
(assert (and (index-in i 4) (index-in i-other 4) (index-in lane-other 32)))
(assert (or (not (= lane lane-other)) (not (= i i-other))))
(assert (= (source-d-row lane i) (source-d-row lane-other i-other)))
(assert (= (source-d-col lane i) (source-d-col lane-other i-other)))
(check-sat)
(pop)

(declare-const total-k Int)
(declare-const group-begin Int)
(declare-const count Int)
(declare-const offset Int)
(define-fun packed-stride () Int (div (+ total-k 1) 2))
(define-fun column () Int (+ group-begin offset))

; Odd starts and short tails use individual guarded nibble loads.
(echo "guarded-tail-read-in-bounds")
(push)
(assert (and (<= 0 group-begin) (<= 0 offset) (<= (+ group-begin count) total-k)))
(assert (and (index-in offset count) (index-in i 8) (< i (- count offset))))
(assert (not (index-in (div (+ column i) 2) packed-stride)))
(check-sat)
(pop)

; A complete aligned register reads exactly four bytes from the same row.
(echo "aligned-register-read-in-bounds")
(push)
(assert (and (<= 0 group-begin) (<= 0 offset) (<= (+ group-begin count) total-k)))
(assert (and (= (mod column 2) 0) (<= (+ offset 8) count) (index-in i 4)))
(assert (not (index-in (+ (div column 2) i) packed-stride)))
(check-sat)
(pop)

; The shape ABI caps count to INT32_MAX / (128 * 15) = 1118481. For A4, the centered
; dot and the correction each have absolute value at most 64 * count.
(echo "int32-accumulator-bound")
(push)
(assert (and (<= 1 count) (<= count 1118481)))
(assert (or (> (* 128 count) 2147483647) (< (- (* 128 count)) (- 2147483648))))
(check-sat)
(pop)
