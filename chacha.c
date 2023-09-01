#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>

#define ASSUME_PADDING 1  /* if true, assume 32 bytes of padding after data (64 for avx512 version) */

void chacha20_in_place_generic(uint8_t* data, size_t len, uint32_t key[8], uint32_t nonce[2]);
void chacha20_in_place_sse(uint8_t* data, size_t len, uint32_t key[8], uint32_t nonce[2]);
void chacha20_in_place_avx2(uint8_t* data, size_t len, uint32_t key[8], uint32_t nonce[2]);
void chacha20_in_place_avx512(uint8_t* data, size_t len, uint32_t key[8], uint32_t nonce[2]);

#define CHACHA_ROUNDS 20
const char* chacha_initial_constant = "expand 32-byte k";
uint32_t rotate_left(uint32_t a, uint32_t count) {
    return (a << count) | (a >> (32 - count));
}

#define QR(a, b, c, d) (			\
        a += b,  d ^= a,  d = rotate_left(d,16),	\
        c += d,  b ^= c,  b = rotate_left(b,12),	\
        a += b,  d ^= a,  d = rotate_left(d, 8),	\
        c += d,  b ^= c,  b = rotate_left(b, 7))

// Adapted from https://en.wikipedia.org/wiki/Salsa20 
void chacha20_in_place_generic(uint8_t* data, size_t len, uint32_t key[8], uint32_t nonce[2]) {
    uint32_t init[16], block[16];
    memcpy(init, chacha_initial_constant, 16);
    memcpy(init + 4, key, 32);
    memcpy(init + 14, nonce, 8);

    uint64_t last = len / 64;
    uint8_t as_u8[64];

    for (uint64_t j = 0; j <= last; ++j) {
        memcpy(init + 12, &j, 8);
        memcpy(block, init, sizeof(block));

        for (int i = 0; i < CHACHA_ROUNDS; i += 2) {
            QR(block[0], block[4], block[ 8], block[12]); // column 0
            QR(block[1], block[5], block[ 9], block[13]); // column 1
            QR(block[2], block[6], block[10], block[14]); // column 2
            QR(block[3], block[7], block[11], block[15]); // column 3

            QR(block[0], block[5], block[10], block[15]); // diagonal 1 (main diagonal)
            QR(block[1], block[6], block[11], block[12]); // diagonal 2
            QR(block[2], block[7], block[ 8], block[13]); // diagonal 3
            QR(block[3], block[4], block[ 9], block[14]); // diagonal 4
        }


        for (int i = 0; i < 16; ++i)
            block[i] += init[i];

        memcpy(as_u8, block, sizeof(block));   // avoid strict aliasing UB 
        for (uint64_t i = 0, k = 64 * j; i < 64 && k < len; ++i, ++k) {
            data[k] ^= as_u8[i];
        }
    }
}

               

__attribute__((always_inline)) __m256i rotate_16_avx2(__m256i a) {
    const __m256i rotate_16 = _mm256_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
                13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    return _mm256_shuffle_epi8(a, rotate_16);
}

__attribute__((always_inline)) __m256i rotate_8_avx2(__m256i a) {
    const __m256i rotate_8 = _mm256_set_epi8(
                14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3,
                14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3);
    return _mm256_shuffle_epi8(a, rotate_8);
}

__attribute__((always_inline)) __m256i rotate_left_avx2(__m256i a, const int count) {
    return _mm256_or_si256(_mm256_slli_epi32(a, count), _mm256_srli_epi32(a, 32 - count));
}

#define QR_COLUMN \
        a = _mm256_add_epi32(a, b),  d = _mm256_xor_si256(a, d),  d = rotate_16_avx2(d),	\
        c = _mm256_add_epi32(c, d),  b = _mm256_xor_si256(b, c),  b = rotate_left_avx2(b, 12),	\
        a = _mm256_add_epi32(a, b),  d = _mm256_xor_si256(a, d),  d = rotate_8_avx2(d),	\
        c = _mm256_add_epi32(c, d),  b = _mm256_xor_si256(b, c),  b = rotate_left_avx2(b, 7);
#define PERM_256(a, imm) \
    (_mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(a), imm)))
#define QR_SHUFFLE \
        b = PERM_256(b, 0x39); \
        c = PERM_256(c, 0x4e); \
        d = PERM_256(d, 0x93);
#define QR_REV_SHUFFLE \
        b = PERM_256(b, 0x93); \
        c = PERM_256(c, 0x4e); \
        d = PERM_256(d, 0x39);

void print_m256(__m256i a) {
    uint32_t k[8];
    _mm256_storeu_si256((__m256i*)k, a);

    for (int i = 0; i < 8; ++i) {
        printf("%x ", k[i]);
    }
    printf("\n");
}
    
void chacha20_in_place_avx2(uint8_t* data, size_t len, uint32_t key[8], uint32_t nonce[2]) {
    if (len <= 128) {
        chacha20_in_place_generic(data, len, key, nonce);
        return;
    }
    
    //  states[0]: [  state1 row1 | state2 row1  ]
    //  states[1]: [  state1 row2 | state2 row2  ]
    //  etc.
    __m256i init[4];
    __m256i a, b, c, d, p1, p2, p3, p4;

    init[0] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) chacha_initial_constant));
    init[1] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) key));
    init[2] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) key + 1));

    uint64_t nonce_u64;
    memcpy(&nonce_u64, nonce, sizeof(nonce_u64));

    init[3] = _mm256_set_epi64x(nonce_u64, 1, nonce_u64, 0);
    __m256i incr_count = _mm256_set_epi64x(0, 2, 0, 2);

    uint8_t* end = data + len;
    int done = 0;

    for (uint8_t* base = data; !done; base += 128) {
        a = init[0];
        b = init[1];
        c = init[2];
        d = init[3];

        for (int i = 0; i < CHACHA_ROUNDS; i += 2) {
            QR_COLUMN
            QR_SHUFFLE
            QR_COLUMN
            QR_REV_SHUFFLE
        }

        a = _mm256_add_epi32(a, init[0]);
        b = _mm256_add_epi32(b, init[1]);
        c = _mm256_add_epi32(c, init[2]);
        d = _mm256_add_epi32(d, init[3]);

        init[3] = _mm256_add_epi64(init[3], incr_count);

        // Order is a little weird, so we need to unpack
        
        p1 = _mm256_permute2x128_si256(a, b, 0x20);
        p2 = _mm256_permute2x128_si256(c, d, 0x20);
        p3 = _mm256_permute2x128_si256(a, b, 0x31);
        p4 = _mm256_permute2x128_si256(c, d, 0x31);

#define DO_XOR(addr, v) \
            _mm256_storeu_si256((__m256i*) (addr), _mm256_xor_si256(v, _mm256_loadu_si256((const __m256i*) (addr))));

        static const void* jump_table_1[] = { 
            &&remain_1_32, &&remain_1_64, &&remain_1_96, &&remain_1_128
        };

        done = base + 128 >= end;
        if (__builtin_expect(done, 0)) {
            int cleanup_idx = (end - base - 1) >> 5;
            if (ASSUME_PADDING) {  // ok for 32-byte over-write
                goto *jump_table_1[cleanup_idx];
            } else {
                // just go byte by byte
                _Alignas(32) uint8_t p[128];
                _mm256_storeu_si256((__m256i*) p, p1);
                _mm256_storeu_si256((__m256i*) p + 1, p2);
                _mm256_storeu_si256((__m256i*) p + 2, p3);
                _mm256_storeu_si256((__m256i*) p + 3, p4);

                for (int i = 0; i < 128 && base < end; ++i, ++base) {
                    *base ^= p[i];
                }

                return;
            }
        }

remain_1_128:
        DO_XOR(base + 96, p4);
remain_1_96:
        DO_XOR(base + 64, p3);
remain_1_64:
        DO_XOR(base + 32, p2);
remain_1_32:
        DO_XOR(base, p1);
    }
}

#ifdef __AVX512F__

__attribute__((always_inline)) __m512i rotate_16_avx512(__m512i a) {
    const __m512i rotate_16 = _mm512_set_epi8(
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
                13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    return _mm512_shuffle_epi8(a, rotate_16);
}

__attribute__((always_inline)) __m512i rotate_8_avx512(__m512i a) {
    const __m512i rotate_8 = _mm512_set_epi8(
                14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3,
                14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3,
                14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3,
                14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3);
    return _mm512_shuffle_epi8(a, rotate_8);
}

#undef DO_XOR
#define DO_XOR(addr, v) \
            _mm512_storeu_si512((__m512i*) (addr), _mm512_xor_si512(v, _mm512_loadu_si512((const __m512i*) (addr))));

#undef QR_COLUMN
#undef PERM_256
#undef QR_SHUFFLE
#undef QR_REV_SHUFFLE

#define QR_COLUMN \
        a = _mm512_add_epi32(a, b),  d = _mm512_xor_si512(a, d),  d = rotate_16_avx512(d),	\
        c = _mm512_add_epi32(c, d),  b = _mm512_xor_si512(b, c),  b = _mm512_rol_epi32(b, 12),	\
        a = _mm512_add_epi32(a, b),  d = _mm512_xor_si512(a, d),  d = rotate_8_avx512(d),	\
        c = _mm512_add_epi32(c, d),  b = _mm512_xor_si512(b, c),  b = _mm512_rol_epi32(b, 7);
#define PERM_512(a, imm) \
    (_mm512_castps_si512(_mm512_permute_ps(_mm512_castsi512_ps(a), imm)))
#define QR_SHUFFLE \
        b = PERM_512(b, 0x39); \
        c = PERM_512(c, 0x4e); \
        d = PERM_512(d, 0x93);
#define QR_REV_SHUFFLE \
        b = PERM_512(b, 0x93); \
        c = PERM_512(c, 0x4e); \
        d = PERM_512(d, 0x39);

void chacha20_in_place_avx512(uint8_t* data, size_t len, uint32_t key[8], uint32_t nonce[2]) {
    if (len <= 4096 /* terribly overkill for anything small */) {
        chacha20_in_place_avx2(data, len, key, nonce);
        return;
    }
    
    //  states[0]: [  state1 row1 | state2 row1 | state3 row1 | state4 row1 ]
    //  etc.
    __m512i init[4];
    __m512i a, b, c, d;

    init[0] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) chacha_initial_constant));
    init[1] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) key));
    init[2] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) key + 1));


    uint64_t nonce_u64;
    memcpy(&nonce_u64, nonce, sizeof(nonce_u64));

    init[3] = _mm512_set_epi64(nonce_u64, 3, nonce_u64, 2, nonce_u64, 1, nonce_u64, 0);
    __m512i incr_count = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);

    uint8_t* end = data + len;
    int done = 0;
    int cleanup_idx = 0;

    for (uint8_t* base = data; !done; base += 256) {
        a = init[0];
        b = init[1];
        c = init[2];
        d = init[3];

        for (int i = 0; i < CHACHA_ROUNDS; i += 2) {
            QR_COLUMN
            QR_SHUFFLE
            QR_COLUMN
            QR_REV_SHUFFLE
        }

        a = _mm512_add_epi32(a, init[0]);
        b = _mm512_add_epi32(b, init[1]);
        c = _mm512_add_epi32(c, init[2]);
        d = _mm512_add_epi32(d, init[3]);

        init[3] = _mm512_add_epi64(init[3], incr_count);

        // Unpack
        
        __m512i px = _mm512_shuffle_i64x2(a, b, 0x88);  // [ 31 | 31 ]
        __m512i py = _mm512_shuffle_i64x2(c, d, 0x88);  // [ 31 | 31 ]
        __m512i pz = _mm512_shuffle_i64x2(a, b, 0xdd);  // [ 42 | 42 ]
        __m512i pw = _mm512_shuffle_i64x2(c, d, 0xdd);  // [ 42 | 42 ]

        __m512i select_ll = _mm512_set_epi64( 13, 12, 9, 8, 5, 4, 1, 0 );
        __m512i select_hh = _mm512_set_epi64( 15, 14, 11, 10, 7, 6, 3, 2 );

        __m512i p1 = _mm512_permutex2var_epi64(px, select_ll, py);
        __m512i p3 = _mm512_permutex2var_epi64(px, select_hh, py);
        __m512i p2 = _mm512_permutex2var_epi64(pz, select_ll, pw);
        __m512i p4 = _mm512_permutex2var_epi64(pz, select_hh, pw);

        static const void* jump_table_2[] = { 
            &&remain_2_64, &&remain_2_128, &&remain_2_192, &&remain_2_256
        };

        done = base + 256 >= end;
        if (__builtin_expect(done, 0)) {
            int cleanup_idx = (end - base -1) >> 6;
            if (ASSUME_PADDING) {  // 64 bytes
                goto *jump_table_2[cleanup_idx];
            } else {
                // just go byte by byte
                _Alignas(64) uint8_t p[256];
                _mm512_storeu_si512((__m512i*) p, p1);
                _mm512_storeu_si512((__m512i*) p + 1, p2);
                _mm512_storeu_si512((__m512i*) p + 2, p3);
                _mm512_storeu_si512((__m512i*) p + 3, p4);

                for (int i = 0; i < 256 && base < end; ++i, ++base) {
                    *base ^= p[i];
                }

                return;
            }
        }

remain_2_256:
        DO_XOR(base + 192, p4);
remain_2_192:
        DO_XOR(base + 128, p3);
remain_2_128:
        DO_XOR(base + 64, p2);
remain_2_64:
        DO_XOR(base, p1);
    }
}

#endif /* AVX512F */

uint32_t test_key[8] = {
    0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c
};

uint32_t test_nonce[2] = {
    0x03020100, 0x07060504
};

uint32_t key_stream[10000] = {
0x89a198f7, 0x69e695f1, 0xfb5f1082, 0x75b70b64, 0xa39d577f, 0x93fc0216, 0x56ac01ec, 0xc1c35af8, 0x7b54a434, 0x41463b73, 0x44c94230, 0x69174900, 0x59bed305, 0xf1531cea, 0x5c151659, 0x1a24e82b, 0x9a8b0038, 0x9435bc26, 0x1744241e, 0x66de8a7c, 0x2695de89, 0x58d98649, 0xe860fb89, 0xbdc92946, 0x1ccb5a9a, 0x56be18c1, 0xa4b3b93e, 0x2ef872a4, 0x78e7a709, 0x2e562b49, 0x880e13f7, 0xc731e0df, 0xf7d4b99d, 0x1599a8c7, 0x50479a1b, 0xc33fb632, 0xe05f2485, 0x5adde354, 0x76f5a597, 0x254006fe, 0x2c04ced3, 0xc5b26a56, 0xdb38b107, 0x693d3e85, 0x96096659, 0xc4c96c54, 0xc7fdeaa6, 0xd740c077, 0xf746af0e, 0x7939ad6d, 0x0c36c5e5, 0x6a161733, 0x944c891c, 0x6a8771a3, 0x2876df94, 0xf2aa4efe, 0x5a7db2cc, 0x7aade0aa, 0xb6d4f9d0, 0x09543bad, 0x52d44687, 0x7a40384d, 0x0000eb6d
};

double cycles_per_rdtsc;    // estimate

void test_impl(void (*impl)(uint8_t*, size_t, uint32_t*, uint32_t*), const char* impl_name) {
    // Correctness test
    // Test for memory overread by putting it at the end of a page

    int is_generic = impl_name[0] == 'g';
    const int TEST_LEN = 16000;
    const int MAX_LEN = is_generic ? 225 : TEST_LEN;

    const int PADDING = (ASSUME_PADDING * ((strcmp(impl_name, "avx2") == 0) ? 33 : 64));
   
    for (int length = 0; length < MAX_LEN; ++length) {
        uint8_t* page = mmap(NULL, 16384, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(page != MAP_FAILED);

        uint8_t* m = page + 16384 - length - 1 - PADDING;

        impl(m, length, test_key, test_nonce);
        fflush(0);
        assert(memcmp(m, key_stream, length) == 0);

        munmap(page, 16384);
    }

    if (is_generic) {
        memset(key_stream, 0, sizeof(key_stream));
        impl((uint8_t*) key_stream, TEST_LEN, test_key, test_nonce);
    }

    // Perf test
    const int LEN = 50000, REPEAT = 100;
    uint8_t* buf = calloc(LEN + PADDING, 1);
    assert(buf);

    uint64_t start = __rdtsc();

    for (int i = 0; i < REPEAT; ++i) {
        impl(buf, LEN, test_key, test_nonce);
    }

    uint64_t end = __rdtsc();

    // prevent optimization
    volatile uint8_t k = buf[0];

    printf("Implementation %s: %f cycles / byte\n", impl_name, (end - start) * cycles_per_rdtsc / (LEN * REPEAT));
    free(buf);
}

void estimate_cycles_per_rdtsc() {
    uint64_t start = __rdtsc();
    asm volatile (
            "mov $100000, %%eax;"
            "1:"
            "dec %%eax;"
            "jnz 1b;"
            ::: "%eax"
            );
    uint64_t end = __rdtsc();

    cycles_per_rdtsc = 100000. / (end - start);
}

int main(int argc, char** argv) {
    estimate_cycles_per_rdtsc();
    
    test_impl(chacha20_in_place_generic, "generic");
    printf("=====\n");
    test_impl(chacha20_in_place_avx2, "avx2");
#ifdef __AVX512F__
    printf("=====\n");
    test_impl(chacha20_in_place_avx512, "avx512");
#endif
}