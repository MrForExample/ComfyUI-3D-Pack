// Prelude for PyTorch cpp binding.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <vector>
#include <stdexcept>
#include <string>

#ifdef SLANG_LLVM
#ifndef SLANG_LLVM_H
#define SLANG_LLVM_H

// TODO(JS): 
// Disable exception declspecs, as not supported on LLVM without some extra options.
// We could enable with `-fms-extensions`
#define SLANG_DISABLE_EXCEPTIONS 1

#ifndef SLANG_PRELUDE_ASSERT
#   ifdef SLANG_PRELUDE_ENABLE_ASSERT
extern "C" void assertFailure(const char* msg);
#       define SLANG_PRELUDE_EXPECT(VALUE, MSG) if(VALUE) {} else assertFailure("assertion failed: '" MSG "'")
#       define SLANG_PRELUDE_ASSERT(VALUE) SLANG_PRELUDE_EXPECT(VALUE, #VALUE)
#   else // SLANG_PRELUDE_ENABLE_ASSERT
#       define SLANG_PRELUDE_EXPECT(VALUE, MSG)
#       define SLANG_PRELUDE_ASSERT(x) 
#   endif // SLANG_PRELUDE_ENABLE_ASSERT
#endif

/*
Taken from stddef.h 
*/

typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
typedef __SIZE_TYPE__ rsize_t;

//typedef __WCHAR_TYPE__ wchar_t;

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#  if !defined(__MINGW32__) && !defined(_MSC_VER)
#    define NULL __null
#  else
#    define NULL 0
#  endif
#else
#  define NULL ((void*)0)
#endif
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std { typedef decltype(nullptr) nullptr_t; }
using ::std::nullptr_t;
#endif
#endif
#undef __need_NULL
#endif /* defined(__need_NULL) */


/*
The following are taken verbatim from stdint.h from Clang in LLVM. Only 8/16/32/64 types are needed. 
*/

// LLVM/Clang types such that we can use LLVM/Clang without headers for C++ output from Slang

#ifdef __INT64_TYPE__
# ifndef __int8_t_defined /* glibc sys/types.h also defines int64_t*/
typedef __INT64_TYPE__ int64_t;
# endif /* __int8_t_defined */
typedef __UINT64_TYPE__ uint64_t;
# define __int_least64_t int64_t
# define __uint_least64_t uint64_t
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
typedef __int_least64_t int_least64_t;
typedef __uint_least64_t uint_least64_t;
typedef __int_least64_t int_fast64_t;
typedef __uint_least64_t uint_fast64_t;
#endif /* __int_least64_t */

#ifdef __INT32_TYPE__

# ifndef __int8_t_defined /* glibc sys/types.h also defines int32_t*/
typedef __INT32_TYPE__ int32_t;
# endif /* __int8_t_defined */

# ifndef __uint32_t_defined  /* more glibc compatibility */
# define __uint32_t_defined
typedef __UINT32_TYPE__ uint32_t;
# endif /* __uint32_t_defined */

# define __int_least32_t int32_t
# define __uint_least32_t uint32_t
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
typedef __int_least32_t int_least32_t;
typedef __uint_least32_t uint_least32_t;
typedef __int_least32_t int_fast32_t;
typedef __uint_least32_t uint_fast32_t;
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int16_t*/
typedef __INT16_TYPE__ int16_t;
#endif /* __int8_t_defined */
typedef __UINT16_TYPE__ uint16_t;
# define __int_least16_t int16_t
# define __uint_least16_t uint16_t
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
typedef __int_least16_t int_least16_t;
typedef __uint_least16_t uint_least16_t;
typedef __int_least16_t int_fast16_t;
typedef __uint_least16_t uint_fast16_t;
#endif /* __int_least16_t */

#ifdef __INT8_TYPE__
#ifndef __int8_t_defined  /* glibc sys/types.h also defines int8_t*/
typedef __INT8_TYPE__ int8_t;
#endif /* __int8_t_defined */
typedef __UINT8_TYPE__ uint8_t;
# define __int_least8_t int8_t
# define __uint_least8_t uint8_t
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
typedef __int_least8_t int_least8_t;
typedef __uint_least8_t uint_least8_t;
typedef __int_least8_t int_fast8_t;
typedef __uint_least8_t uint_fast8_t;
#endif /* __int_least8_t */

/* prevent glibc sys/types.h from defining conflicting types */
#ifndef __int8_t_defined
# define __int8_t_defined
#endif /* __int8_t_defined */

/* C99 7.18.1.4 Integer types capable of holding object pointers.
 */
#define __stdint_join3(a,b,c) a ## b ## c

#ifndef _INTPTR_T
#ifndef __intptr_t_defined
typedef __INTPTR_TYPE__ intptr_t;
#define __intptr_t_defined
#define _INTPTR_T
#endif
#endif

#ifndef _UINTPTR_T
typedef __UINTPTR_TYPE__ uintptr_t;
#define _UINTPTR_T
#endif

/* C99 7.18.1.5 Greatest-width integer types.
 */
typedef __INTMAX_TYPE__  intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;

/* C99 7.18.4 Macros for minimum-width integer constants.
 *
 * The standard requires that integer constant macros be defined for all the
 * minimum-width types defined above. As 8-, 16-, 32-, and 64-bit minimum-width
 * types are required, the corresponding integer constant macros are defined
 * here. This implementation also defines minimum-width types for every other
 * integer width that the target implements, so corresponding macros are
 * defined below, too.
 *
 * These macros are defined using the same successive-shrinking approach as
 * the type definitions above. It is likewise important that macros are defined
 * in order of decending width.
 *
 * Note that C++ should not check __STDC_CONSTANT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#define __int_c_join(a, b) a ## b
#define __int_c(v, suffix) __int_c_join(v, suffix)
#define __uint_c(v, suffix) __int_c_join(v##U, suffix)

#ifdef __INT64_TYPE__
# ifdef __INT64_C_SUFFIX__
#  define __int64_c_suffix __INT64_C_SUFFIX__
# else
#  undef __int64_c_suffix
# endif /* __INT64_C_SUFFIX__ */
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
# ifdef __int64_c_suffix
#  define INT64_C(v) __int_c(v, __int64_c_suffix)
#  define UINT64_C(v) __uint_c(v, __int64_c_suffix)
# else
#  define INT64_C(v) v
#  define UINT64_C(v) v ## U
# endif /* __int64_c_suffix */
#endif /* __int_least64_t */


#ifdef __INT32_TYPE__
# ifdef __INT32_C_SUFFIX__
#  define __int32_c_suffix __INT32_C_SUFFIX__
#else
#  undef __int32_c_suffix
# endif /* __INT32_C_SUFFIX__ */
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
# ifdef __int32_c_suffix
#  define INT32_C(v) __int_c(v, __int32_c_suffix)
#  define UINT32_C(v) __uint_c(v, __int32_c_suffix)
# else
#  define INT32_C(v) v
#  define UINT32_C(v) v ## U
# endif /* __int32_c_suffix */
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
# ifdef __INT16_C_SUFFIX__
#  define __int16_c_suffix __INT16_C_SUFFIX__
#else
#  undef __int16_c_suffix
# endif /* __INT16_C_SUFFIX__ */
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
# ifdef __int16_c_suffix
#  define INT16_C(v) __int_c(v, __int16_c_suffix)
#  define UINT16_C(v) __uint_c(v, __int16_c_suffix)
# else
#  define INT16_C(v) v
#  define UINT16_C(v) v ## U
# endif /* __int16_c_suffix */
#endif /* __int_least16_t */


#ifdef __INT8_TYPE__
# ifdef __INT8_C_SUFFIX__
#  define __int8_c_suffix __INT8_C_SUFFIX__
#else
#  undef  __int8_c_suffix
# endif /* __INT8_C_SUFFIX__ */
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
# ifdef __int8_c_suffix
#  define INT8_C(v) __int_c(v, __int8_c_suffix)
#  define UINT8_C(v) __uint_c(v, __int8_c_suffix)
# else
#  define INT8_C(v) v
#  define UINT8_C(v) v ## U
# endif /* __int8_c_suffix */
#endif /* __int_least8_t */

/* C99 7.18.2.1 Limits of exact-width integer types.
 * C99 7.18.2.2 Limits of minimum-width integer types.
 * C99 7.18.2.3 Limits of fastest minimum-width integer types.
 *
 * The presence of limit macros are completely optional in C99.  This
 * implementation defines limits for all of the types (exact- and
 * minimum-width) that it defines above, using the limits of the minimum-width
 * type for any types that do not have exact-width representations.
 *
 * As in the type definitions, this section takes an approach of
 * successive-shrinking to determine which limits to use for the standard (8,
 * 16, 32, 64) bit widths when they don't have exact representations. It is
 * therefore important that the definitions be kept in order of decending
 * widths.
 *
 * Note that C++ should not check __STDC_LIMIT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#ifdef __INT64_TYPE__
# define INT64_MAX           INT64_C( 9223372036854775807)
# define INT64_MIN         (-INT64_C( 9223372036854775807)-1)
# define UINT64_MAX         UINT64_C(18446744073709551615)
# define __INT_LEAST64_MIN   INT64_MIN
# define __INT_LEAST64_MAX   INT64_MAX
# define __UINT_LEAST64_MAX UINT64_MAX
#endif /* __INT64_TYPE__ */

#ifdef __INT_LEAST64_MIN
# define INT_LEAST64_MIN   __INT_LEAST64_MIN
# define INT_LEAST64_MAX   __INT_LEAST64_MAX
# define UINT_LEAST64_MAX __UINT_LEAST64_MAX
# define INT_FAST64_MIN    __INT_LEAST64_MIN
# define INT_FAST64_MAX    __INT_LEAST64_MAX
# define UINT_FAST64_MAX  __UINT_LEAST64_MAX
#endif /* __INT_LEAST64_MIN */

#ifdef __INT32_TYPE__
# define INT32_MAX           INT32_C(2147483647)
# define INT32_MIN         (-INT32_C(2147483647)-1)
# define UINT32_MAX         UINT32_C(4294967295)
# define __INT_LEAST32_MIN   INT32_MIN
# define __INT_LEAST32_MAX   INT32_MAX
# define __UINT_LEAST32_MAX UINT32_MAX
#endif /* __INT32_TYPE__ */

#ifdef __INT_LEAST32_MIN
# define INT_LEAST32_MIN   __INT_LEAST32_MIN
# define INT_LEAST32_MAX   __INT_LEAST32_MAX
# define UINT_LEAST32_MAX __UINT_LEAST32_MAX
# define INT_FAST32_MIN    __INT_LEAST32_MIN
# define INT_FAST32_MAX    __INT_LEAST32_MAX
# define UINT_FAST32_MAX  __UINT_LEAST32_MAX
#endif /* __INT_LEAST32_MIN */

#ifdef __INT16_TYPE__
#define INT16_MAX            INT16_C(32767)
#define INT16_MIN          (-INT16_C(32767)-1)
#define UINT16_MAX          UINT16_C(65535)
# define __INT_LEAST16_MIN   INT16_MIN
# define __INT_LEAST16_MAX   INT16_MAX
# define __UINT_LEAST16_MAX UINT16_MAX
#endif /* __INT16_TYPE__ */

#ifdef __INT_LEAST16_MIN
# define INT_LEAST16_MIN   __INT_LEAST16_MIN
# define INT_LEAST16_MAX   __INT_LEAST16_MAX
# define UINT_LEAST16_MAX __UINT_LEAST16_MAX
# define INT_FAST16_MIN    __INT_LEAST16_MIN
# define INT_FAST16_MAX    __INT_LEAST16_MAX
# define UINT_FAST16_MAX  __UINT_LEAST16_MAX
#endif /* __INT_LEAST16_MIN */


#ifdef __INT8_TYPE__
# define INT8_MAX            INT8_C(127)
# define INT8_MIN          (-INT8_C(127)-1)
# define UINT8_MAX          UINT8_C(255)
# define __INT_LEAST8_MIN    INT8_MIN
# define __INT_LEAST8_MAX    INT8_MAX
# define __UINT_LEAST8_MAX  UINT8_MAX
#endif /* __INT8_TYPE__ */

#ifdef __INT_LEAST8_MIN
# define INT_LEAST8_MIN   __INT_LEAST8_MIN
# define INT_LEAST8_MAX   __INT_LEAST8_MAX
# define UINT_LEAST8_MAX __UINT_LEAST8_MAX
# define INT_FAST8_MIN    __INT_LEAST8_MIN
# define INT_FAST8_MAX    __INT_LEAST8_MAX
# define UINT_FAST8_MAX  __UINT_LEAST8_MAX
#endif /* __INT_LEAST8_MIN */

/* Some utility macros */
#define  __INTN_MIN(n)  __stdint_join3( INT, n, _MIN)
#define  __INTN_MAX(n)  __stdint_join3( INT, n, _MAX)
#define __UINTN_MAX(n)  __stdint_join3(UINT, n, _MAX)
#define  __INTN_C(n, v) __stdint_join3( INT, n, _C(v))
#define __UINTN_C(n, v) __stdint_join3(UINT, n, _C(v))

/* C99 7.18.2.4 Limits of integer types capable of holding object pointers. */
/* C99 7.18.3 Limits of other integer types. */

#define  INTPTR_MIN  (-__INTPTR_MAX__-1)
#define  INTPTR_MAX    __INTPTR_MAX__
#define UINTPTR_MAX   __UINTPTR_MAX__
#define PTRDIFF_MIN (-__PTRDIFF_MAX__-1)
#define PTRDIFF_MAX   __PTRDIFF_MAX__
#define    SIZE_MAX      __SIZE_MAX__

/* ISO9899:2011 7.20 (C11 Annex K): Define RSIZE_MAX if __STDC_WANT_LIB_EXT1__
 * is enabled. */
#if defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1
#define   RSIZE_MAX            (SIZE_MAX >> 1)
#endif

/* C99 7.18.2.5 Limits of greatest-width integer types. */
#define  INTMAX_MIN (-__INTMAX_MAX__-1)
#define  INTMAX_MAX   __INTMAX_MAX__
#define UINTMAX_MAX  __UINTMAX_MAX__

/* C99 7.18.3 Limits of other integer types. */
#define SIG_ATOMIC_MIN __INTN_MIN(__SIG_ATOMIC_WIDTH__)
#define SIG_ATOMIC_MAX __INTN_MAX(__SIG_ATOMIC_WIDTH__)
#ifdef __WINT_UNSIGNED__
# define WINT_MIN       __UINTN_C(__WINT_WIDTH__, 0)
# define WINT_MAX       __UINTN_MAX(__WINT_WIDTH__)
#else
# define WINT_MIN       __INTN_MIN(__WINT_WIDTH__)
# define WINT_MAX       __INTN_MAX(__WINT_WIDTH__)
#endif

#ifndef WCHAR_MAX
# define WCHAR_MAX __WCHAR_MAX__
#endif
#ifndef WCHAR_MIN
# if __WCHAR_MAX__ == __INTN_MAX(__WCHAR_WIDTH__)
#  define WCHAR_MIN __INTN_MIN(__WCHAR_WIDTH__)
# else
#  define WCHAR_MIN __UINTN_C(__WCHAR_WIDTH__, 0)
# endif
#endif

/* 7.18.4.2 Macros for greatest-width integer constants. */
#define  INTMAX_C(v) __int_c(v,  __INTMAX_C_SUFFIX__)
#define UINTMAX_C(v) __int_c(v, __UINTMAX_C_SUFFIX__)


#endif // SLANG_LLVM_H



#else // SLANG_LLVM
#   if SLANG_GCC_FAMILY && __GNUC__ < 6
#       include <cmath>
#       define SLANG_PRELUDE_STD std::
#   else
#       include <math.h>
#       define SLANG_PRELUDE_STD
#   endif

#   include <assert.h>
#   include <stdlib.h>
#   include <string.h>
#   include <stdint.h>
#endif // SLANG_LLVM

#ifndef SLANG_CORE_STRING_H
#define SLANG_CORE_STRING_H

#include <string.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#ifndef SLANG_CORE_SMART_POINTER_H
#define SLANG_CORE_SMART_POINTER_H

#ifndef SLANG_CORE_COMMON_H
#define SLANG_CORE_COMMON_H

#ifndef SLANG_H
#define SLANG_H

/** \file slang.h

The Slang API provides services to compile, reflect, and specialize code
written in the Slang shading language.
*/

/*
The following section attempts to detect the compiler and version in use.

If an application defines `SLANG_COMPILER` before including this header,
they take responsibility for setting any compiler-dependent macros
used later in the file.

Most applications should not need to touch this section.
*/
#ifndef SLANG_COMPILER
#    define SLANG_COMPILER

/*
Compiler defines, see http://sourceforge.net/p/predef/wiki/Compilers/
NOTE that SLANG_VC holds the compiler version - not just 1 or 0
*/
#    if defined(_MSC_VER)
#        if _MSC_VER >= 1900
#            define SLANG_VC 14
#        elif _MSC_VER >= 1800
#            define SLANG_VC 12
#        elif _MSC_VER >= 1700
#            define SLANG_VC 11
#        elif _MSC_VER >= 1600
#            define SLANG_VC 10
#        elif _MSC_VER >= 1500
#            define SLANG_VC 9
#        else
#            error "unknown version of Visual C++ compiler"
#        endif
#    elif defined(__clang__)
#        define SLANG_CLANG 1
#    elif defined(__SNC__)
#        define SLANG_SNC 1
#    elif defined(__ghs__)
#        define SLANG_GHS 1
#    elif defined(__GNUC__) /* note: __clang__, __SNC__, or __ghs__ imply __GNUC__ */
#        define SLANG_GCC 1
#    else
#        error "unknown compiler"
#    endif
/*
Any compilers not detected by the above logic are now now explicitly zeroed out.
*/
#    ifndef SLANG_VC
#        define SLANG_VC 0
#    endif
#    ifndef SLANG_CLANG
#        define SLANG_CLANG 0
#    endif
#    ifndef SLANG_SNC
#        define SLANG_SNC 0
#    endif
#    ifndef SLANG_GHS
#        define SLANG_GHS 0
#    endif
#    ifndef SLANG_GCC
#        define SLANG_GCC 0
#    endif
#endif /* SLANG_COMPILER */

/*
The following section attempts to detect the target platform being compiled for.

If an application defines `SLANG_PLATFORM` before including this header,
they take responsibility for setting any compiler-dependent macros
used later in the file.

Most applications should not need to touch this section.
*/
#ifndef SLANG_PLATFORM
#    define SLANG_PLATFORM
/**
Operating system defines, see http://sourceforge.net/p/predef/wiki/OperatingSystems/
*/
#    if defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_PARTITION_APP
#        define SLANG_WINRT 1 /* Windows Runtime, either on Windows RT or Windows 8 */
#    elif defined(XBOXONE)
#        define SLANG_XBOXONE 1
#    elif defined(_WIN64) /* note: XBOXONE implies _WIN64 */
#        define SLANG_WIN64 1
#    elif defined(_M_PPC)
#        define SLANG_X360 1
#    elif defined(_WIN32) /* note: _M_PPC implies _WIN32 */
#        define SLANG_WIN32 1
#    elif defined(__ANDROID__)
#        define SLANG_ANDROID 1
#    elif defined(__linux__) || defined(__CYGWIN__) /* note: __ANDROID__ implies __linux__ */
#        define SLANG_LINUX 1
#    elif defined(__APPLE__)
#        include "TargetConditionals.h"
#        if TARGET_OS_MAC
#            define SLANG_OSX 1
#        else
#            define SLANG_IOS 1
#        endif
#    elif defined(__CELLOS_LV2__)
#        define SLANG_PS3 1
#    elif defined(__ORBIS__)
#        define SLANG_PS4 1
#    elif defined(__SNC__) && defined(__arm__)
#        define SLANG_PSP2 1
#    elif defined(__ghs__)
#        define SLANG_WIIU 1
#    else
#        error "unknown target platform"
#    endif
/*
Any platforms not detected by the above logic are now now explicitly zeroed out.
*/
#    ifndef SLANG_WINRT
#        define SLANG_WINRT 0
#    endif
#    ifndef SLANG_XBOXONE
#        define SLANG_XBOXONE 0
#    endif
#    ifndef SLANG_WIN64
#        define SLANG_WIN64 0
#    endif
#    ifndef SLANG_X360
#        define SLANG_X360 0
#    endif
#    ifndef SLANG_WIN32
#        define SLANG_WIN32 0
#    endif
#    ifndef SLANG_ANDROID
#        define SLANG_ANDROID 0
#    endif
#    ifndef SLANG_LINUX
#        define SLANG_LINUX 0
#    endif
#    ifndef SLANG_IOS
#        define SLANG_IOS 0
#    endif
#    ifndef SLANG_OSX
#        define SLANG_OSX 0
#    endif
#    ifndef SLANG_PS3
#        define SLANG_PS3 0
#    endif
#    ifndef SLANG_PS4
#        define SLANG_PS4 0
#    endif
#    ifndef SLANG_PSP2
#        define SLANG_PSP2 0
#    endif
#    ifndef SLANG_WIIU
#        define SLANG_WIIU 0
#    endif
#endif /* SLANG_PLATFORM */

/* Shorthands for "families" of compilers/platforms */
#define SLANG_GCC_FAMILY (SLANG_CLANG || SLANG_SNC || SLANG_GHS || SLANG_GCC)
#define SLANG_WINDOWS_FAMILY (SLANG_WINRT || SLANG_WIN32 || SLANG_WIN64)
#define SLANG_MICROSOFT_FAMILY (SLANG_XBOXONE || SLANG_X360 || SLANG_WINDOWS_FAMILY)
#define SLANG_LINUX_FAMILY (SLANG_LINUX || SLANG_ANDROID)
#define SLANG_APPLE_FAMILY (SLANG_IOS || SLANG_OSX)                  /* equivalent to #if __APPLE__ */
#define SLANG_UNIX_FAMILY (SLANG_LINUX_FAMILY || SLANG_APPLE_FAMILY) /* shortcut for unix/posix platforms */

/* Macros concerning DirectX */
#if !defined(SLANG_CONFIG_DX_ON_VK) || !SLANG_CONFIG_DX_ON_VK
#    define SLANG_ENABLE_DXVK 0
#    define SLANG_ENABLE_VKD3D 0
#else
#    define SLANG_ENABLE_DXVK 1
#    define SLANG_ENABLE_VKD3D 1
#endif

#if SLANG_WINDOWS_FAMILY
#    define SLANG_ENABLE_DIRECTX 1
#    define SLANG_ENABLE_DXGI_DEBUG 1
#    define SLANG_ENABLE_DXBC_SUPPORT 1
#    define SLANG_ENABLE_PIX 1
#elif SLANG_LINUX_FAMILY
#    define SLANG_ENABLE_DIRECTX (SLANG_ENABLE_DXVK || SLANG_ENABLE_VKD3D)
#    define SLANG_ENABLE_DXGI_DEBUG 0
#    define SLANG_ENABLE_DXBC_SUPPORT 0
#    define SLANG_ENABLE_PIX 0
#else
#    define SLANG_ENABLE_DIRECTX 0
#    define SLANG_ENABLE_DXGI_DEBUG 0
#    define SLANG_ENABLE_DXBC_SUPPORT 0
#    define SLANG_ENABLE_PIX 0
#endif

/* Macro for declaring if a method is no throw. Should be set before the return parameter. */
#ifndef SLANG_NO_THROW
#   if SLANG_WINDOWS_FAMILY && !defined(SLANG_DISABLE_EXCEPTIONS)
#       define SLANG_NO_THROW __declspec(nothrow)
#   endif
#endif
#ifndef SLANG_NO_THROW
#   define SLANG_NO_THROW
#endif

/* The `SLANG_STDCALL` and `SLANG_MCALL` defines are used to set the calling
convention for interface methods.
*/
#ifndef SLANG_STDCALL
#   if SLANG_MICROSOFT_FAMILY
#       define SLANG_STDCALL __stdcall
#   else
#       define SLANG_STDCALL
#   endif
#endif
#ifndef SLANG_MCALL
#   define SLANG_MCALL SLANG_STDCALL
#endif


#if !defined(SLANG_STATIC) && !defined(SLANG_DYNAMIC)
    #define SLANG_DYNAMIC
#endif

#if defined(_MSC_VER)
#   define SLANG_DLL_EXPORT __declspec(dllexport)
#else
#   if 0 && __GNUC__ >= 4
// Didn't work on latest gcc on linux.. so disable for now
// https://gcc.gnu.org/wiki/Visibility
#       define SLANG_DLL_EXPORT __attribute__ ((dllexport))
#   else
#       define SLANG_DLL_EXPORT __attribute__((__visibility__("default")))
#   endif
#endif

#if defined(SLANG_DYNAMIC)
#   if defined(_MSC_VER)
#       ifdef SLANG_DYNAMIC_EXPORT
#           define SLANG_API SLANG_DLL_EXPORT
#       else
#           define SLANG_API __declspec(dllimport)
#       endif
#   else
        // TODO: need to consider compiler capabilities
//#     ifdef SLANG_DYNAMIC_EXPORT
#       define SLANG_API SLANG_DLL_EXPORT 
//#     endif
#   endif
#endif

#ifndef SLANG_API
#   define SLANG_API
#endif

// GCC Specific
#if SLANG_GCC_FAMILY

#	define SLANG_NO_INLINE __attribute__((noinline))
#	define SLANG_FORCE_INLINE inline __attribute__((always_inline))
#   define SLANG_BREAKPOINT(id) __builtin_trap();
#	define SLANG_ALIGN_OF(T)	__alignof__(T)

// Use the builtin directly so we don't need to have an include of stddef.h
#   define SLANG_OFFSET_OF(T, ELEMENT) __builtin_offsetof(T, ELEMENT) 
#endif // SLANG_GCC_FAMILY

#ifndef SLANG_OFFSET_OF
#   define SLANG_OFFSET_OF(T, ELEMENT) (size_t(&((T*)1)->ELEMENT) - 1)
#endif

// Microsoft VC specific
#if SLANG_MICROSOFT_FAMILY
#	define SLANG_NO_INLINE __declspec(noinline)
#	define SLANG_FORCE_INLINE __forceinline
#	define SLANG_BREAKPOINT(id) __debugbreak();
#	define SLANG_ALIGN_OF(T) __alignof(T)

#   define SLANG_INT64(x) (x##i64)
#   define SLANG_UINT64(x) (x##ui64)
#endif // SLANG_MICROSOFT_FAMILY

#ifndef SLANG_FORCE_INLINE
#	define SLANG_FORCE_INLINE inline
#endif
#ifndef SLANG_NO_INLINE
#	define SLANG_NO_INLINE
#endif

#ifndef SLANG_COMPILE_TIME_ASSERT
#   define SLANG_COMPILE_TIME_ASSERT(x) static_assert(x)
#endif

#ifndef SLANG_OFFSET_OF
#	define SLANG_OFFSET_OF(X, Y) offsetof(X, Y)
#endif

#ifndef SLANG_BREAKPOINT
// Make it crash with a write to 0!
#   define SLANG_BREAKPOINT(id) (*((int*)0) = int(id));
#endif

// Use for getting the amount of members of a standard C array.
// Use 0[x] here to catch the case where x has an overloaded subscript operator
#define SLANG_COUNT_OF(x) (SlangSSizeT(sizeof(x)/sizeof(0[x])))
/// SLANG_INLINE exists to have a way to inline consistent with SLANG_ALWAYS_INLINE
#define SLANG_INLINE inline

// If explicilty disabled and not set, set to not available
#if !defined(SLANG_HAS_EXCEPTIONS) && defined(SLANG_DISABLE_EXCEPTIONS)
#   define SLANG_HAS_EXCEPTIONS 0
#endif

// If not set, the default is exceptions are available
#ifndef SLANG_HAS_EXCEPTIONS
#   define SLANG_HAS_EXCEPTIONS 1
#endif

// Other defines
#define SLANG_STRINGIZE_HELPER(X) #X
#define SLANG_STRINGIZE(X) SLANG_STRINGIZE_HELPER(X)

#define SLANG_CONCAT_HELPER(X, Y) X##Y
#define SLANG_CONCAT(X, Y) SLANG_CONCAT_HELPER(X, Y)

#ifndef SLANG_UNUSED
#	define SLANG_UNUSED(v) (void)v;
#endif

// Used for doing constant literals
#ifndef SLANG_INT64
#	define SLANG_INT64(x) (x##ll)
#endif
#ifndef SLANG_UINT64
#	define SLANG_UINT64(x) (x##ull)
#endif


#ifdef __cplusplus
#   define SLANG_EXTERN_C extern "C"
#else
#   define SLANG_EXTERN_C
#endif

#ifdef __cplusplus
// C++ specific macros
// Clang
#if SLANG_CLANG
#    if (__clang_major__*10 + __clang_minor__) >= 33
#       define SLANG_HAS_MOVE_SEMANTICS 1
#       define SLANG_HAS_ENUM_CLASS 1
#       define SLANG_OVERRIDE override
#    endif

// Gcc
#elif SLANG_GCC_FAMILY
// Check for C++11
#		if (__cplusplus >= 201103L)
#			if (__GNUC__ * 100 + __GNUC_MINOR__) >= 405
#				define SLANG_HAS_MOVE_SEMANTICS 1
#			endif
#			if (__GNUC__ * 100 + __GNUC_MINOR__) >= 406
#				define SLANG_HAS_ENUM_CLASS 1
#			endif
#			if (__GNUC__ * 100 + __GNUC_MINOR__) >= 407
#				define SLANG_OVERRIDE override
#			endif
#		endif

// TODO(JS): Not used in previous code. Left here as may be useful on some other version. 
// #define SLANG_RETURN_NEVER __attribute__((__noreturn__))

#       define SLANG_RETURN_NEVER [[noreturn]]

#	endif // SLANG_GCC_FAMILY

// Visual Studio

#	if SLANG_VC
// C4481: nonstandard extension used: override specifier 'override'
#		if _MSC_VER < 1700
#			pragma warning(disable : 4481)
#		endif
#		define SLANG_OVERRIDE	override
#		if _MSC_VER >= 1600
#			define SLANG_HAS_MOVE_SEMANTICS 1
#		endif
#	    if _MSC_VER >= 1700
#		    define SLANG_HAS_ENUM_CLASS 1
#       endif

#   define SLANG_RETURN_NEVER __declspec(noreturn)

#   endif // SLANG_VC

// Set non set
#   ifndef SLANG_OVERRIDE
#	    define SLANG_OVERRIDE
#   endif
#   ifndef SLANG_HAS_ENUM_CLASS
#	    define SLANG_HAS_ENUM_CLASS 0
#   endif
#   ifndef SLANG_HAS_MOVE_SEMANTICS
#	    define SLANG_HAS_MOVE_SEMANTICS 0
#   endif

#endif // __cplusplus

#ifndef SLANG_RETURN_NEVER
#   define SLANG_RETURN_NEVER [[noreturn]]
#endif // SLANG_RETURN_NEVER

/* Macros for detecting processor */
#if defined(_M_ARM) || defined(__ARM_EABI__)
// This is special case for nVidia tegra
#   define SLANG_PROCESSOR_ARM 1
#elif defined(__i386__) || defined(_M_IX86)
#   define SLANG_PROCESSOR_X86 1
#elif defined(_M_AMD64) || defined(_M_X64) || defined(__amd64) || defined(__x86_64)
#   define SLANG_PROCESSOR_X86_64 1
#elif defined(_PPC_) || defined(__ppc__) || defined(__POWERPC__) || defined(_M_PPC)
#   if defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__) || defined(__64BIT__) || defined(_LP64) || defined(__LP64__)
#       define SLANG_PROCESSOR_POWER_PC_64 1
#   else
#       define SLANG_PROCESSOR_POWER_PC 1
#   endif
#elif defined(__arm__)
#   define SLANG_PROCESSOR_ARM 1
#elif defined(_M_ARM64) || defined(__aarch64__)
#   define SLANG_PROCESSOR_ARM_64 1
#endif 

#ifndef SLANG_PROCESSOR_ARM
#   define SLANG_PROCESSOR_ARM 0
#endif

#ifndef SLANG_PROCESSOR_ARM_64
#   define SLANG_PROCESSOR_ARM_64 0
#endif

#ifndef SLANG_PROCESSOR_X86
#   define SLANG_PROCESSOR_X86 0
#endif

#ifndef SLANG_PROCESSOR_X86_64
#   define SLANG_PROCESSOR_X86_64 0
#endif

#ifndef SLANG_PROCESSOR_POWER_PC
#   define SLANG_PROCESSOR_POWER_PC 0
#endif

#ifndef SLANG_PROCESSOR_POWER_PC_64
#   define SLANG_PROCESSOR_POWER_PC_64 0
#endif

// Processor families

#define SLANG_PROCESSOR_FAMILY_X86 (SLANG_PROCESSOR_X86_64 | SLANG_PROCESSOR_X86)
#define SLANG_PROCESSOR_FAMILY_ARM (SLANG_PROCESSOR_ARM | SLANG_PROCESSOR_ARM_64)
#define SLANG_PROCESSOR_FAMILY_POWER_PC (SLANG_PROCESSOR_POWER_PC_64 | SLANG_PROCESSOR_POWER_PC)

// Pointer size
#define SLANG_PTR_IS_64 (SLANG_PROCESSOR_ARM_64 | SLANG_PROCESSOR_X86_64 | SLANG_PROCESSOR_POWER_PC_64)
#define SLANG_PTR_IS_32 (SLANG_PTR_IS_64 ^ 1)

// Processor features
#if SLANG_PROCESSOR_FAMILY_X86
#   define SLANG_LITTLE_ENDIAN 1
#   define SLANG_UNALIGNED_ACCESS 1
#elif SLANG_PROCESSOR_FAMILY_ARM
#   if defined(__ARMEB__)
#       define SLANG_BIG_ENDIAN 1
#   else
#       define SLANG_LITTLE_ENDIAN 1
#   endif
#elif SLANG_PROCESSOR_FAMILY_POWER_PC
#       define SLANG_BIG_ENDIAN 1
#endif

#ifndef SLANG_LITTLE_ENDIAN
#   define SLANG_LITTLE_ENDIAN 0
#endif

#ifndef SLANG_BIG_ENDIAN
#   define SLANG_BIG_ENDIAN 0
#endif

#ifndef SLANG_UNALIGNED_ACCESS
#   define SLANG_UNALIGNED_ACCESS 0
#endif

// One endianess must be set
#if ((SLANG_BIG_ENDIAN | SLANG_LITTLE_ENDIAN) == 0)
#   error "Couldn't determine endianess"
#endif

#ifndef  SLANG_NO_INTTYPES
#include <inttypes.h>
#endif // ! SLANG_NO_INTTYPES

#ifndef  SLANG_NO_STDDEF
#include <stddef.h>
#endif // ! SLANG_NO_STDDEF

#ifdef __cplusplus
extern "C"
{
#endif
    /*!
    @mainpage Introduction

    API Reference: slang.h

    @file slang.h
    */

    typedef uint32_t    SlangUInt32;
    typedef int32_t     SlangInt32;

    // Use SLANG_PTR_ macros to determine SlangInt/SlangUInt types.
    // This is used over say using size_t/ptrdiff_t/intptr_t/uintptr_t, because on some targets, these types are distinct from
    // their uint_t/int_t equivalents and so produce ambiguity with function overloading.
    //
    // SlangSizeT is helpful as on some compilers size_t is distinct from a regular integer type and so overloading doesn't work.
    // Casting to SlangSizeT works around this.
#if SLANG_PTR_IS_64
    typedef int64_t    SlangInt;
    typedef uint64_t   SlangUInt;

    typedef int64_t    SlangSSizeT;
    typedef uint64_t   SlangSizeT;
#else
    typedef int32_t    SlangInt;
    typedef uint32_t   SlangUInt;

    typedef int32_t    SlangSSizeT;
    typedef uint32_t   SlangSizeT;
#endif

    typedef bool SlangBool;

    
    /*!
    @brief Severity of a diagnostic generated by the compiler.
    Values come from the enum below, with higher values representing more severe
    conditions, and all values >= SLANG_SEVERITY_ERROR indicating compilation
    failure.
    */
    typedef int SlangSeverityIntegral;
    enum SlangSeverity : SlangSeverityIntegral
    {
        SLANG_SEVERITY_DISABLED = 0, /**< A message that is disabled, filtered out. */
        SLANG_SEVERITY_NOTE,         /**< An informative message. */
        SLANG_SEVERITY_WARNING,      /**< A warning, which indicates a possible proble. */
        SLANG_SEVERITY_ERROR,        /**< An error, indicating that compilation failed. */
        SLANG_SEVERITY_FATAL,        /**< An unrecoverable error, which forced compilation to abort. */
        SLANG_SEVERITY_INTERNAL,     /**< An internal error, indicating a logic error in the compiler. */
    };

    typedef int SlangDiagnosticFlags;
    enum
    {
        SLANG_DIAGNOSTIC_FLAG_VERBOSE_PATHS = 0x01,
        SLANG_DIAGNOSTIC_FLAG_TREAT_WARNINGS_AS_ERRORS = 0x02
    };

    typedef int SlangBindableResourceIntegral;
    enum SlangBindableResourceType : SlangBindableResourceIntegral
    {
        SLANG_NON_BINDABLE = 0,
        SLANG_TEXTURE,
        SLANG_SAMPLER,
        SLANG_UNIFORM_BUFFER,
        SLANG_STORAGE_BUFFER,
    };

    /* NOTE! To keep binary compatibility care is needed with this enum!

    * To add value, only add at the bottom (before COUNT_OF) 
    * To remove a value, add _DEPRECATED as a suffix, but leave in the list
    
    This will make the enum values stable, and compatible with libraries that might not use the latest
    enum values.
    */
    typedef int SlangCompileTargetIntegral;
    enum SlangCompileTarget : SlangCompileTargetIntegral
    {
        SLANG_TARGET_UNKNOWN,
        SLANG_TARGET_NONE,
        SLANG_GLSL,
        SLANG_GLSL_VULKAN_DEPRECATED,              //< deprecated and removed: just use `SLANG_GLSL`.
        SLANG_GLSL_VULKAN_ONE_DESC_DEPRECATED,     //< deprecated and removed.
        SLANG_HLSL,
        SLANG_SPIRV,
        SLANG_SPIRV_ASM,
        SLANG_DXBC,
        SLANG_DXBC_ASM,
        SLANG_DXIL,
        SLANG_DXIL_ASM,
        SLANG_C_SOURCE,                 ///< The C language
        SLANG_CPP_SOURCE,               ///< C++ code for shader kernels.
        SLANG_HOST_EXECUTABLE,          ///< Standalone binary executable (for hosting CPU/OS)
        SLANG_SHADER_SHARED_LIBRARY,    ///< A shared library/Dll for shader kernels (for hosting CPU/OS)
        SLANG_SHADER_HOST_CALLABLE,     ///< A CPU target that makes the compiled shader code available to be run immediately
        SLANG_CUDA_SOURCE,              ///< Cuda source
        SLANG_PTX,                      ///< PTX
        SLANG_CUDA_OBJECT_CODE,         ///< Object code that contains CUDA functions.
        SLANG_OBJECT_CODE,              ///< Object code that can be used for later linking
        SLANG_HOST_CPP_SOURCE,          ///< C++ code for host library or executable.
        SLANG_HOST_HOST_CALLABLE,       ///< Host callable host code (ie non kernel/shader) 
        SLANG_CPP_PYTORCH_BINDING,      ///< C++ PyTorch binding code.
        SLANG_METAL,                    ///< Metal shading language
        SLANG_METAL_LIB,                ///< Metal library
        SLANG_METAL_LIB_ASM,            ///< Metal library assembly
        SLANG_HOST_SHARED_LIBRARY,      ///< A shared library/Dll for host code (for hosting CPU/OS)
        SLANG_TARGET_COUNT_OF,
    };

    /* A "container format" describes the way that the outputs
    for multiple files, entry points, targets, etc. should be
    combined into a single artifact for output. */
    typedef int SlangContainerFormatIntegral;
    enum SlangContainerFormat : SlangContainerFormatIntegral
    {
        /* Don't generate a container. */
        SLANG_CONTAINER_FORMAT_NONE,

        /* Generate a container in the `.slang-module` format,
        which includes reflection information, compiled kernels, etc. */
        SLANG_CONTAINER_FORMAT_SLANG_MODULE,
    };

    typedef int SlangPassThroughIntegral;
    enum SlangPassThrough : SlangPassThroughIntegral
    {
        SLANG_PASS_THROUGH_NONE,
        SLANG_PASS_THROUGH_FXC,
        SLANG_PASS_THROUGH_DXC,
        SLANG_PASS_THROUGH_GLSLANG,
        SLANG_PASS_THROUGH_SPIRV_DIS,
        SLANG_PASS_THROUGH_CLANG,                   ///< Clang C/C++ compiler 
        SLANG_PASS_THROUGH_VISUAL_STUDIO,           ///< Visual studio C/C++ compiler
        SLANG_PASS_THROUGH_GCC,                     ///< GCC C/C++ compiler
        SLANG_PASS_THROUGH_GENERIC_C_CPP,           ///< Generic C or C++ compiler, which is decided by the source type
        SLANG_PASS_THROUGH_NVRTC,                   ///< NVRTC Cuda compiler
        SLANG_PASS_THROUGH_LLVM,                    ///< LLVM 'compiler' - includes LLVM and Clang
        SLANG_PASS_THROUGH_SPIRV_OPT,               ///< SPIRV-opt
        SLANG_PASS_THROUGH_METAL,                   ///< Metal compiler
        SLANG_PASS_THROUGH_COUNT_OF,
    };

    /* Defines an archive type used to holds a 'file system' type structure. */
    typedef int SlangArchiveTypeIntegral;
    enum SlangArchiveType : SlangArchiveTypeIntegral
    {
        SLANG_ARCHIVE_TYPE_UNDEFINED,
        SLANG_ARCHIVE_TYPE_ZIP,
        SLANG_ARCHIVE_TYPE_RIFF,                ///< Riff container with no compression
        SLANG_ARCHIVE_TYPE_RIFF_DEFLATE,
        SLANG_ARCHIVE_TYPE_RIFF_LZ4,
        SLANG_ARCHIVE_TYPE_COUNT_OF,
    };

    /*!
    Flags to control compilation behavior.
    */
    typedef unsigned int SlangCompileFlags;
    enum
    {
        /* Do as little mangling of names as possible, to try to preserve original names */
        SLANG_COMPILE_FLAG_NO_MANGLING          = 1 << 3,

        /* Skip code generation step, just check the code and generate layout */
        SLANG_COMPILE_FLAG_NO_CODEGEN           = 1 << 4,

        /* Obfuscate shader names on release products */
        SLANG_COMPILE_FLAG_OBFUSCATE = 1 << 5,

        /* Deprecated flags: kept around to allow existing applications to
        compile. Note that the relevant features will still be left in
        their default state. */
        SLANG_COMPILE_FLAG_NO_CHECKING          = 0,
        SLANG_COMPILE_FLAG_SPLIT_MIXED_TYPES    = 0,
    };

    /*!
    @brief Flags to control code generation behavior of a compilation target */
    typedef unsigned int SlangTargetFlags;
    enum 
    {
        /* When compiling for a D3D Shader Model 5.1 or higher target, allocate
           distinct register spaces for parameter blocks.

           @deprecated This behavior is now enabled unconditionally.
        */
        SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES = 1 << 4,

        /* When set, will generate target code that contains all entrypoints defined
           in the input source or specified via the `spAddEntryPoint` function in a
           single output module (library/source file).
        */
        SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM = 1 << 8,

        /* When set, will dump out the IR between intermediate compilation steps.*/
        SLANG_TARGET_FLAG_DUMP_IR = 1 << 9,

        /* When set, will generate SPIRV directly rather than via glslang. */
        SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY = 1 << 10,
    };
    constexpr static SlangTargetFlags kDefaultTargetFlags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;

    /*!
    @brief Options to control floating-point precision guarantees for a target.
    */
    typedef unsigned int SlangFloatingPointModeIntegral;
    enum SlangFloatingPointMode : SlangFloatingPointModeIntegral
    {
        SLANG_FLOATING_POINT_MODE_DEFAULT = 0,
        SLANG_FLOATING_POINT_MODE_FAST,
        SLANG_FLOATING_POINT_MODE_PRECISE,
    };

    /*!
    @brief Options to control emission of `#line` directives
    */
    typedef unsigned int SlangLineDirectiveModeIntegral;
    enum SlangLineDirectiveMode : SlangLineDirectiveModeIntegral
    {
        SLANG_LINE_DIRECTIVE_MODE_DEFAULT = 0,  /**< Default behavior: pick behavior base on target. */
        SLANG_LINE_DIRECTIVE_MODE_NONE,         /**< Don't emit line directives at all. */
        SLANG_LINE_DIRECTIVE_MODE_STANDARD,     /**< Emit standard C-style `#line` directives. */
        SLANG_LINE_DIRECTIVE_MODE_GLSL,         /**< Emit GLSL-style directives with file *number* instead of name */
        SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP,   /**< Use a source map to track line mappings (ie no #line will appear in emitting source) */
    };

    typedef int SlangSourceLanguageIntegral;
    enum SlangSourceLanguage : SlangSourceLanguageIntegral
    {
        SLANG_SOURCE_LANGUAGE_UNKNOWN,
        SLANG_SOURCE_LANGUAGE_SLANG,
        SLANG_SOURCE_LANGUAGE_HLSL,
        SLANG_SOURCE_LANGUAGE_GLSL,
        SLANG_SOURCE_LANGUAGE_C,
        SLANG_SOURCE_LANGUAGE_CPP,
        SLANG_SOURCE_LANGUAGE_CUDA,
        SLANG_SOURCE_LANGUAGE_SPIRV,
        SLANG_SOURCE_LANGUAGE_METAL,
        SLANG_SOURCE_LANGUAGE_COUNT_OF,
    };

    typedef unsigned int SlangProfileIDIntegral;
    enum SlangProfileID : SlangProfileIDIntegral
    {
        SLANG_PROFILE_UNKNOWN,
    };


    typedef SlangInt32 SlangCapabilityIDIntegral;
    enum SlangCapabilityID : SlangCapabilityIDIntegral
    {
        SLANG_CAPABILITY_UNKNOWN = 0,
    };

    typedef unsigned int SlangMatrixLayoutModeIntegral;
    enum SlangMatrixLayoutMode : SlangMatrixLayoutModeIntegral
    {
        SLANG_MATRIX_LAYOUT_MODE_UNKNOWN = 0,
        SLANG_MATRIX_LAYOUT_ROW_MAJOR,
        SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
    };

    typedef SlangUInt32 SlangStageIntegral;
    enum SlangStage : SlangStageIntegral
    {
        SLANG_STAGE_NONE,
        SLANG_STAGE_VERTEX,
        SLANG_STAGE_HULL,
        SLANG_STAGE_DOMAIN,
        SLANG_STAGE_GEOMETRY,
        SLANG_STAGE_FRAGMENT,
        SLANG_STAGE_COMPUTE,
        SLANG_STAGE_RAY_GENERATION,
        SLANG_STAGE_INTERSECTION,
        SLANG_STAGE_ANY_HIT,
        SLANG_STAGE_CLOSEST_HIT,
        SLANG_STAGE_MISS,
        SLANG_STAGE_CALLABLE,
        SLANG_STAGE_MESH,
        SLANG_STAGE_AMPLIFICATION,

        // alias:
        SLANG_STAGE_PIXEL = SLANG_STAGE_FRAGMENT,
    };

    typedef SlangUInt32 SlangDebugInfoLevelIntegral;
    enum SlangDebugInfoLevel : SlangDebugInfoLevelIntegral
    {
        SLANG_DEBUG_INFO_LEVEL_NONE = 0,    /**< Don't emit debug information at all. */
        SLANG_DEBUG_INFO_LEVEL_MINIMAL,     /**< Emit as little debug information as possible, while still supporting stack trackes. */
        SLANG_DEBUG_INFO_LEVEL_STANDARD,    /**< Emit whatever is the standard level of debug information for each target. */
        SLANG_DEBUG_INFO_LEVEL_MAXIMAL,     /**< Emit as much debug infromation as possible for each target. */
        
    };

    /* Describes the debugging information format produced during a compilation. */
    typedef SlangUInt32 SlangDebugInfoFormatIntegral;
    enum SlangDebugInfoFormat : SlangDebugInfoFormatIntegral
    {
        SLANG_DEBUG_INFO_FORMAT_DEFAULT,         ///< Use the default debugging format for the target 
        SLANG_DEBUG_INFO_FORMAT_C7,              ///< CodeView C7 format (typically means debugging infomation is embedded in the binary)
        SLANG_DEBUG_INFO_FORMAT_PDB,             ///< Program database
        
        SLANG_DEBUG_INFO_FORMAT_STABS,          ///< Stabs
        SLANG_DEBUG_INFO_FORMAT_COFF,           ///< COFF debug info
        SLANG_DEBUG_INFO_FORMAT_DWARF,          ///< DWARF debug info (we may want to support specifying the version)

        SLANG_DEBUG_INFO_FORMAT_COUNT_OF,
    };

    typedef SlangUInt32 SlangOptimizationLevelIntegral;
    enum SlangOptimizationLevel : SlangOptimizationLevelIntegral
    {
        SLANG_OPTIMIZATION_LEVEL_NONE = 0,  /**< Don't optimize at all. */
        SLANG_OPTIMIZATION_LEVEL_DEFAULT,   /**< Default optimization level: balance code quality and compilation time. */
        SLANG_OPTIMIZATION_LEVEL_HIGH,      /**< Optimize aggressively. */
        SLANG_OPTIMIZATION_LEVEL_MAXIMAL,   /**< Include optimizations that may take a very long time, or may involve severe space-vs-speed tradeoffs */
    };

    // All compiler option names supported by Slang.
    namespace slang
    {
        enum class CompilerOptionName
        {
            MacroDefine,        // stringValue0: macro name;  stringValue1: macro value
            DepFile,
            EntryPointName,
            Specialize,
            Help,
            HelpStyle,
            Include,            // stringValue: additional include path.
            Language,
            MatrixLayoutColumn, // bool
            MatrixLayoutRow,    // bool
            ZeroInitialize,     // bool
            IgnoreCapabilities, // bool
            RestrictiveCapabilityCheck, // bool
            ModuleName,         // stringValue0: module name.
            Output,
            Profile,            // intValue0: profile
            Stage,              // intValue0: stage
            Target,             // intValue0: CodeGenTarget
            Version,
            WarningsAsErrors,   // stringValue0: "all" or comma separated list of warning codes or names.
            DisableWarnings,    // stringValue0: comma separated list of warning codes or names.
            EnableWarning,      // stringValue0: warning code or name.
            DisableWarning,     // stringValue0: warning code or name.
            DumpWarningDiagnostics,
            InputFilesRemain,
            EmitIr,                // bool
            ReportDownstreamTime,  // bool
            ReportPerfBenchmark,   // bool
            SkipSPIRVValidation,   // bool
            SourceEmbedStyle,
            SourceEmbedName,
            SourceEmbedLanguage,
            DisableShortCircuit,   // bool
            MinimumSlangOptimization, // bool
            DisableNonEssentialValidations, // bool
            DisableSourceMap,       // bool
            UnscopedEnum,           // bool
            PreserveParameters,       // bool: preserve all resource parameters in the output code.

            // Target

            Capability,                 // intValue0: CapabilityName
            DefaultImageFormatUnknown,  // bool
            DisableDynamicDispatch,     // bool
            DisableSpecialization,      // bool
            FloatingPointMode,          // intValue0: FloatingPointMode
            DebugInformation,           // intValue0: DebugInfoLevel
            LineDirectiveMode,
            Optimization,               // intValue0: OptimizationLevel
            Obfuscate,                  // bool

            VulkanBindShift,            // intValue0 (higher 8 bits): kind; intValue0(lower bits): set; intValue1: shift
            VulkanBindGlobals,          // intValue0: index; intValue1: set
            VulkanInvertY,              // bool
            VulkanUseDxPositionW,       // bool
            VulkanUseEntryPointName,    // bool
            VulkanUseGLLayout,          // bool
            VulkanEmitReflection,       // bool

            GLSLForceScalarLayout,      // bool
            EnableEffectAnnotations,    // bool

            EmitSpirvViaGLSL,           // bool
            EmitSpirvDirectly,          // bool
            SPIRVCoreGrammarJSON,       // stringValue0: json path
            IncompleteLibrary,          // bool, when set, will not issue an error when the linked program has unresolved extern function symbols.

            // Downstream

            CompilerPath,
            DefaultDownstreamCompiler,
            DownstreamArgs,             // stringValue0: downstream compiler name. stringValue1: argument list, one per line.
            PassThrough,

            // Repro

            DumpRepro,
            DumpReproOnError,
            ExtractRepro,
            LoadRepro,
            LoadReproDirectory,
            ReproFallbackDirectory,

            // Debugging

            DumpAst,
            DumpIntermediatePrefix,
            DumpIntermediates,      // bool
            DumpIr,                 // bool
            DumpIrIds,
            PreprocessorOutput,
            OutputIncludes,
            ReproFileSystem,
            SerialIr,               // bool
            SkipCodeGen,            // bool
            ValidateIr,             // bool
            VerbosePaths,
            VerifyDebugSerialIr,
            NoCodeGen,              // Not used.

            // Experimental

            FileSystem,
            Heterogeneous,
            NoMangle,
            NoHLSLBinding,
            NoHLSLPackConstantBufferElements,
            ValidateUniformity,
            AllowGLSL,

            // Internal

            ArchiveType,
            CompileStdLib,
            Doc,
            IrCompression,
            LoadStdLib,
            ReferenceModule,
            SaveStdLib,
            SaveStdLibBinSource,
            TrackLiveness,
            LoopInversion,              // bool, enable loop inversion optimization

            // Deprecated
            ParameterBlocksUseRegisterSpaces,

            CountOfParsableOptions,

            // Used in parsed options only.
            DebugInformationFormat,     // intValue0: DebugInfoFormat
            VulkanBindShiftAll,         // intValue0: kind; intValue1: shift
            GenerateWholeProgram,       // bool
            UseUpToDateBinaryModule,    // bool, when set, will only load
                                        // precompiled modules if it is up-to-date with its source.

            CountOf,
        };

        enum class CompilerOptionValueKind
        {
            Int,
            String
        };

        struct CompilerOptionValue
        {
            CompilerOptionValueKind kind = CompilerOptionValueKind::Int;
            int32_t intValue0 = 0;
            int32_t intValue1 = 0;
            const char* stringValue0 = nullptr;
            const char* stringValue1 = nullptr;
        };

        struct CompilerOptionEntry
        {
            CompilerOptionName name;
            CompilerOptionValue value;
        };
    }

    /** A result code for a Slang API operation.

    This type is generally compatible with the Windows API `HRESULT` type. In particular, negative values indicate
    failure results, while zero or positive results indicate success.

    In general, Slang APIs always return a zero result on success, unless documented otherwise. Strictly speaking
    a negative value indicates an error, a positive (or 0) value indicates success. This can be tested for with the macros
    SLANG_SUCCEEDED(x) or SLANG_FAILED(x).

    It can represent if the call was successful or not. It can also specify in an extensible manner what facility
    produced the result (as the integral 'facility') as well as what caused it (as an integral 'code').
    Under the covers SlangResult is represented as a int32_t.

    SlangResult is designed to be compatible with COM HRESULT.

    It's layout in bits is as follows

    Severity | Facility | Code
    ---------|----------|-----
    31       |    30-16 | 15-0

    Severity - 1 fail, 0 is success - as SlangResult is signed 32 bits, means negative number indicates failure.
    Facility is where the error originated from. Code is the code specific to the facility.

    Result codes have the following styles,
    1) SLANG_name
    2) SLANG_s_f_name
    3) SLANG_s_name

    where s is S for success, E for error
    f is the short version of the facility name

    Style 1 is reserved for SLANG_OK and SLANG_FAIL as they are so commonly used.

    It is acceptable to expand 'f' to a longer name to differentiate a name or drop if unique without it.
    ie for a facility 'DRIVER' it might make sense to have an error of the form SLANG_E_DRIVER_OUT_OF_MEMORY
    */

    typedef int32_t SlangResult;

    //! Use to test if a result was failure. Never use result != SLANG_OK to test for failure, as there may be successful codes != SLANG_OK.
#define SLANG_FAILED(status) ((status) < 0)
    //! Use to test if a result succeeded. Never use result == SLANG_OK to test for success, as will detect other successful codes as a failure.
#define SLANG_SUCCEEDED(status) ((status) >= 0)

    //! Get the facility the result is associated with
#define SLANG_GET_RESULT_FACILITY(r)    ((int32_t)(((r) >> 16) & 0x7fff))
    //! Get the result code for the facility
#define SLANG_GET_RESULT_CODE(r)        ((int32_t)((r) & 0xffff))

#define SLANG_MAKE_ERROR(fac, code)        ((((int32_t)(fac)) << 16) | ((int32_t)(code)) | int32_t(0x80000000))
#define SLANG_MAKE_SUCCESS(fac, code)    ((((int32_t)(fac)) << 16) | ((int32_t)(code)))

    /*************************** Facilities ************************************/

    //! Facilities compatible with windows COM - only use if known code is compatible
#define SLANG_FACILITY_WIN_GENERAL      0
#define SLANG_FACILITY_WIN_INTERFACE    4
#define SLANG_FACILITY_WIN_API          7

    //! Base facility -> so as to not clash with HRESULT values (values in 0x200 range do not appear used)
#define SLANG_FACILITY_BASE         0x200

    /*! Facilities numbers must be unique across a project to make the resulting result a unique number.
    It can be useful to have a consistent short name for a facility, as used in the name prefix */
#define SLANG_FACILITY_CORE             SLANG_FACILITY_BASE
    /* Facility for codes, that are not uniquely defined/protected. Can be used to pass back a specific error without requiring system wide facility uniqueness. Codes
    should never be part of a public API. */
#define SLANG_FACILITY_INTERNAL         SLANG_FACILITY_BASE + 1

    /// Base for external facilities. Facilities should be unique across modules.
#define SLANG_FACILITY_EXTERNAL_BASE 0x210

    /* ************************ Win COM compatible Results ******************************/
    // https://msdn.microsoft.com/en-us/library/windows/desktop/aa378137(v=vs.85).aspx

    //! SLANG_OK indicates success, and is equivalent to SLANG_MAKE_SUCCESS(SLANG_FACILITY_WIN_GENERAL, 0)
#define SLANG_OK                          0
    //! SLANG_FAIL is the generic failure code - meaning a serious error occurred and the call couldn't complete
#define SLANG_FAIL                          SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_GENERAL, 0x4005)

#define SLANG_MAKE_WIN_GENERAL_ERROR(code)  SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_GENERAL, code)

    //! Functionality is not implemented
#define SLANG_E_NOT_IMPLEMENTED             SLANG_MAKE_WIN_GENERAL_ERROR(0x4001)
    //! Interface not be found
#define SLANG_E_NO_INTERFACE                SLANG_MAKE_WIN_GENERAL_ERROR(0x4002)
    //! Operation was aborted (did not correctly complete)
#define SLANG_E_ABORT                       SLANG_MAKE_WIN_GENERAL_ERROR(0x4004) 

    //! Indicates that a handle passed in as parameter to a method is invalid.
#define SLANG_E_INVALID_HANDLE              SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_API, 6)
    //! Indicates that an argument passed in as parameter to a method is invalid.
#define SLANG_E_INVALID_ARG                 SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_API, 0x57)
    //! Operation could not complete - ran out of memory
#define SLANG_E_OUT_OF_MEMORY               SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_API, 0xe)

    /* *************************** other Results **************************************/

#define SLANG_MAKE_CORE_ERROR(code)         SLANG_MAKE_ERROR(SLANG_FACILITY_CORE, code)

    // Supplied buffer is too small to be able to complete
#define SLANG_E_BUFFER_TOO_SMALL            SLANG_MAKE_CORE_ERROR(1)
    //! Used to identify a Result that has yet to be initialized.
    //! It defaults to failure such that if used incorrectly will fail, as similar in concept to using an uninitialized variable.
#define SLANG_E_UNINITIALIZED               SLANG_MAKE_CORE_ERROR(2)
    //! Returned from an async method meaning the output is invalid (thus an error), but a result for the request is pending, and will be returned on a subsequent call with the async handle.
#define SLANG_E_PENDING                     SLANG_MAKE_CORE_ERROR(3)
    //! Indicates a file/resource could not be opened
#define SLANG_E_CANNOT_OPEN                 SLANG_MAKE_CORE_ERROR(4)
    //! Indicates a file/resource could not be found
#define SLANG_E_NOT_FOUND                   SLANG_MAKE_CORE_ERROR(5)
    //! An unhandled internal failure (typically from unhandled exception)
#define SLANG_E_INTERNAL_FAIL               SLANG_MAKE_CORE_ERROR(6)
    //! Could not complete because some underlying feature (hardware or software) was not available 
#define SLANG_E_NOT_AVAILABLE               SLANG_MAKE_CORE_ERROR(7)
        //! Could not complete because the operation times out. 
#define SLANG_E_TIME_OUT                    SLANG_MAKE_CORE_ERROR(8)

    /** A "Universally Unique Identifier" (UUID)

    The Slang API uses UUIDs to identify interfaces when
    using `queryInterface`.

    This type is compatible with the `GUID` type defined
    by the Component Object Model (COM), but Slang is
    not dependent on COM.
    */
    struct SlangUUID
    {
        uint32_t data1;
        uint16_t data2;
        uint16_t data3;
        uint8_t  data4[8];
    };

// Place at the start of an interface with the guid.
// Guid should be specified as SLANG_COM_INTERFACE(0x00000000, 0x0000, 0x0000, { 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46 })
// NOTE: it's the typical guid struct definition, without the surrounding {}
// It is not necessary to use the multiple parameters (we can wrap in parens), but this is simple.
#define SLANG_COM_INTERFACE(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7) \
    public: \
    SLANG_FORCE_INLINE constexpr static SlangUUID getTypeGuid() \
    { \
        return { a, b, c, d0, d1, d2, d3, d4, d5, d6, d7 }; \
    }

// Sometimes it's useful to associate a guid with a class to identify it. This macro can used for this,
// and the guid extracted via the getTypeGuid() function defined in the type
#define SLANG_CLASS_GUID(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7) \
    SLANG_FORCE_INLINE constexpr static SlangUUID getTypeGuid() \
    { \
        return { a, b, c, d0, d1, d2, d3, d4, d5, d6, d7 }; \
    }

// Helper to fill in pairs of GUIDs and return pointers. This ensures that the
// type of the GUID passed matches the pointer type, and that it is derived
// from ISlangUnknown,
// TODO(c++20): would is_derived_from be more appropriate here for private inheritance of ISlangUnknown?
//
// with     : void createFoo(SlangUUID, void**);
//            Slang::ComPtr<Bar> myBar;
// call with: createFoo(SLANG_IID_PPV_ARGS(myBar.writeRef()))
// to call  : createFoo(Bar::getTypeGuid(), (void**)(myBar.writeRef()))
#define SLANG_IID_PPV_ARGS(ppType) \
    std::decay_t<decltype(**(ppType))>::getTypeGuid(), \
    ((void)[]{static_assert(std::is_base_of_v<ISlangUnknown, std::decay_t<decltype(**(ppType))>>);}, reinterpret_cast<void**>(ppType))


    /** Base interface for components exchanged through the API.

    This interface definition is compatible with the COM `IUnknown`,
    and uses the same UUID, but Slang does not require applications
    to use or initialize COM.
    */
    struct ISlangUnknown
    {
        SLANG_COM_INTERFACE(0x00000000, 0x0000, 0x0000, { 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46 })

        virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject) = 0;
        virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() = 0;
        virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() = 0;

        /*
        Inline methods are provided to allow the above operations to be called
        using their traditional COM names/signatures:
        */
        SlangResult QueryInterface(struct _GUID const& uuid, void** outObject) { return queryInterface(*(SlangUUID const*)&uuid, outObject); }
        uint32_t AddRef() { return addRef(); }
        uint32_t Release() { return release(); }
    };
    #define SLANG_UUID_ISlangUnknown ISlangUnknown::getTypeGuid()


    /* An interface to provide a mechanism to cast, that doesn't require ref counting
    and doesn't have to return a pointer to a ISlangUnknown derived class */
    class ISlangCastable : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(0x87ede0e1, 0x4852, 0x44b0, { 0x8b, 0xf2, 0xcb, 0x31, 0x87, 0x4d, 0xe2, 0x39 });

            /// Can be used to cast to interfaces without reference counting. 
            /// Also provides access to internal implementations, when they provide a guid
            /// Can simulate a 'generated' interface as long as kept in scope by cast from. 
        virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) = 0;
    };

    class ISlangClonable : public ISlangCastable
    {
        SLANG_COM_INTERFACE(0x1ec36168, 0xe9f4, 0x430d, { 0xbb, 0x17, 0x4, 0x8a, 0x80, 0x46, 0xb3, 0x1f });

            /// Note the use of guid is for the desired interface/object.
            /// The object is returned *not* ref counted. Any type that can implements the interface, 
            /// derives from ICastable, and so (not withstanding some other issue) will always return
            /// an ICastable interface which other interfaces/types are accessible from via castAs
        SLANG_NO_THROW virtual void* SLANG_MCALL clone(const SlangUUID& guid) = 0;
    };

    /** A "blob" of binary data.

    This interface definition is compatible with the `ID3DBlob` and `ID3D10Blob` interfaces.
    */
    struct ISlangBlob : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(0x8BA5FB08, 0x5195, 0x40e2, { 0xAC, 0x58, 0x0D, 0x98, 0x9C, 0x3A, 0x01, 0x02 })

        virtual SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() = 0;
        virtual SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() = 0;
    };
    #define SLANG_UUID_ISlangBlob ISlangBlob::getTypeGuid()

    /* Can be requested from ISlangCastable cast to indicate the contained chars are null terminated.  
    */
    struct SlangTerminatedChars
    {
        SLANG_CLASS_GUID(0xbe0db1a8, 0x3594, 0x4603, { 0xa7, 0x8b, 0xc4, 0x86, 0x84, 0x30, 0xdf, 0xbb });
        operator const char*() const { return chars; }
        char chars[1];
    };

    /** A (real or virtual) file system.

    Slang can make use of this interface whenever it would otherwise try to load files
    from disk, allowing applications to hook and/or override filesystem access from
    the compiler.

    It is the responsibility of 
    the caller of any method that returns a ISlangBlob to release the blob when it is no 
    longer used (using 'release').
    */

    struct ISlangFileSystem : public ISlangCastable
    {
        SLANG_COM_INTERFACE(0x003A09FC, 0x3A4D, 0x4BA0, { 0xAD, 0x60, 0x1F, 0xD8, 0x63, 0xA9, 0x15, 0xAB })

        /** Load a file from `path` and return a blob of its contents
        @param path The path to load from, as a null-terminated UTF-8 string.
        @param outBlob A destination pointer to receive the blob of the file contents.
        @returns A `SlangResult` to indicate success or failure in loading the file.

        NOTE! This is a *binary* load - the blob should contain the exact same bytes
        as are found in the backing file. 

        If load is successful, the implementation should create a blob to hold
        the file's content, store it to `outBlob`, and return 0.
        If the load fails, the implementation should return a failure status
        (any negative value will do).
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(
            char const*     path,
            ISlangBlob** outBlob) = 0;
    };
    #define SLANG_UUID_ISlangFileSystem ISlangFileSystem::getTypeGuid()


    typedef void(*SlangFuncPtr)(void);

    /** 
    (DEPRECATED) ISlangSharedLibrary
    */
    struct ISlangSharedLibrary_Dep1: public ISlangUnknown
    {
        SLANG_COM_INTERFACE( 0x9c9d5bc5, 0xeb61, 0x496f,{ 0x80, 0xd7, 0xd1, 0x47, 0xc4, 0xa2, 0x37, 0x30 })

        virtual SLANG_NO_THROW void* SLANG_MCALL findSymbolAddressByName(char const* name) = 0;
    };
    #define SLANG_UUID_ISlangSharedLibrary_Dep1 ISlangSharedLibrary_Dep1::getTypeGuid()

    /** An interface that can be used to encapsulate access to a shared library. An implementation
    does not have to implement the library as a shared library
    */
    struct ISlangSharedLibrary : public ISlangCastable
    {
        SLANG_COM_INTERFACE(0x70dbc7c4, 0xdc3b, 0x4a07, { 0xae, 0x7e, 0x75, 0x2a, 0xf6, 0xa8, 0x15, 0x55 })

        /** Get a function by name. If the library is unloaded will only return nullptr.
        @param name The name of the function
        @return The function pointer related to the name or nullptr if not found
        */
        SLANG_FORCE_INLINE SlangFuncPtr findFuncByName(char const* name) { return (SlangFuncPtr)findSymbolAddressByName(name); }

        /** Get a symbol by name. If the library is unloaded will only return nullptr.
        @param name The name of the symbol
        @return The pointer related to the name or nullptr if not found
        */
        virtual SLANG_NO_THROW void* SLANG_MCALL findSymbolAddressByName(char const* name) = 0;
    };
    #define SLANG_UUID_ISlangSharedLibrary ISlangSharedLibrary::getTypeGuid()

    struct ISlangSharedLibraryLoader: public ISlangUnknown
    {
        SLANG_COM_INTERFACE(0x6264ab2b, 0xa3e8, 0x4a06, { 0x97, 0xf1, 0x49, 0xbc, 0x2d, 0x2a, 0xb1, 0x4d })

            /** Load a shared library. In typical usage the library name should *not* contain any platform
            specific elements. For example on windows a dll name should *not* be passed with a '.dll' extension,
            and similarly on linux a shared library should *not* be passed with the 'lib' prefix and '.so' extension
            @path path The unadorned filename and/or path for the shared library
            @ param sharedLibraryOut Holds the shared library if successfully loaded */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadSharedLibrary(
            const char*     path,
            ISlangSharedLibrary** sharedLibraryOut) = 0;
    };
    #define SLANG_UUID_ISlangSharedLibraryLoader ISlangSharedLibraryLoader::getTypeGuid()
    
    /* Type that identifies how a path should be interpreted */
    typedef unsigned int SlangPathTypeIntegral;
    enum SlangPathType : SlangPathTypeIntegral
    {
        SLANG_PATH_TYPE_DIRECTORY,      /**< Path specified specifies a directory. */
        SLANG_PATH_TYPE_FILE,           /**< Path specified is to a file. */
    };

    /* Callback to enumerate the contents of of a directory in a ISlangFileSystemExt.
    The name is the name of a file system object (directory/file) in the specified path (ie it is without a path) */
    typedef void (*FileSystemContentsCallBack)(SlangPathType pathType, const char* name, void* userData);

    /* Determines how paths map to files on the OS file system */
    enum class OSPathKind : uint8_t
    {
        None,                ///< Paths do not map to the file system
        Direct,              ///< Paths map directly to the file system
        OperatingSystem,     ///< Only paths gained via PathKind::OperatingSystem map to the operating system file system
    };

    /* Used to determine what kind of path is required from an input path */
    enum class PathKind
    {
            /// Given a path, returns a simplified version of that path.  
            /// This typically means removing '..' and/or '.' from the path.
            /// A simplified path must point to the same object as the original.
        Simplified,             

            /// Given a path, returns a 'canonical path' to the item. 
            /// This may be the operating system 'canonical path' that is the unique path to the item.
            /// 
            /// If the item exists the returned canonical path should always be usable to access the item.
            /// 
            /// If the item the path specifies doesn't exist, the canonical path may not be returnable
            /// or be a path simplification.             
            /// Not all file systems support canonical paths.
        Canonical,

            /// Given a path returns a path such that it is suitable to be displayed to the user.
            /// 
            /// For example if the file system is a zip file - it might include the path to the zip
            /// container as well as the path to the specific file.
            /// 
            /// NOTE! The display path won't necessarily work on the file system to access the item
        Display,

            /// Get the path to the item on the *operating system* file system, if available.
        OperatingSystem,

        CountOf,
    };

    /** An extended file system abstraction.
    
    Implementing and using this interface over ISlangFileSystem gives much more control over how paths
    are managed, as well as how it is determined if two files 'are the same'.

    All paths as input char*, or output as ISlangBlobs are always encoded as UTF-8 strings.
    Blobs that contain strings are always zero terminated.
    */
    struct ISlangFileSystemExt : public ISlangFileSystem
    {
        SLANG_COM_INTERFACE(0x5fb632d2, 0x979d, 0x4481, { 0x9f, 0xee, 0x66, 0x3c, 0x3f, 0x14, 0x49, 0xe1 })

        /** Get a uniqueIdentity which uniquely identifies an object of the file system.
           
        Given a path, returns a 'uniqueIdentity' which ideally is the same value for the same object on the file system.

        The uniqueIdentity is used to compare if two paths are the same - which amongst other things allows Slang to
        cache source contents internally. It is also used for #pragma once functionality.

        A *requirement* is for any implementation is that two paths can only return the same uniqueIdentity if the
        contents of the two files are *identical*. If an implementation breaks this constraint it can produce incorrect compilation.
        If an implementation cannot *strictly* identify *the same* files, this will only have an effect on #pragma once behavior.

        The string for the uniqueIdentity is held zero terminated in the ISlangBlob of outUniqueIdentity.
   
        Note that there are many ways a uniqueIdentity may be generated for a file. For example it could be the
        'canonical path' - assuming it is available and unambiguous for a file system. Another possible mechanism
        could be to store the filename combined with the file date time to uniquely identify it.
     
        The client must ensure the blob be released when no longer used, otherwise memory will leak.

        NOTE! Ideally this method would be called 'getPathUniqueIdentity' but for historical reasons and
        backward compatibility it's name remains with 'File' even though an implementation should be made to work
        with directories too.

        @param path
        @param outUniqueIdentity
        @returns A `SlangResult` to indicate success or failure getting the uniqueIdentity.
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getFileUniqueIdentity(
            const char* path,
            ISlangBlob** outUniqueIdentity) = 0;

        /** Calculate a path combining the 'fromPath' with 'path'

        The client must ensure the blob be released when no longer used, otherwise memory will leak.

        @param fromPathType How to interpret the from path - as a file or a directory.
        @param fromPath The from path. 
        @param path Path to be determined relative to the fromPath
        @param pathOut Holds the string which is the relative path. The string is held in the blob zero terminated.  
        @returns A `SlangResult` to indicate success or failure in loading the file.
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(
            SlangPathType fromPathType,
            const char* fromPath,
            const char* path,
            ISlangBlob** pathOut) = 0;          
            
        /** Gets the type of path that path is on the file system. 
        @param path
        @param pathTypeOut
        @returns SLANG_OK if located and type is known, else an error. SLANG_E_NOT_FOUND if not found.
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getPathType(
            const char* path, 
            SlangPathType* pathTypeOut) = 0;

        /** Get a path based on the kind.

        @param kind The kind of path wanted
        @param path The input path
        @param outPath The output path held in a blob
        @returns SLANG_OK if successfully simplified the path (SLANG_E_NOT_IMPLEMENTED if not implemented, or some other error code)
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getPath(
            PathKind kind,
            const char* path,
            ISlangBlob** outPath) = 0;

        /** Clears any cached information */
        virtual SLANG_NO_THROW void SLANG_MCALL clearCache() = 0;

        /** Enumerate the contents of the path
        
        Note that for normal Slang operation it isn't necessary to enumerate contents this can return SLANG_E_NOT_IMPLEMENTED.
        
        @param The path to enumerate
        @param callback This callback is called for each entry in the path. 
        @param userData This is passed to the callback
        @returns SLANG_OK if successful 
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(
            const char* path,
            FileSystemContentsCallBack callback,
            void* userData) = 0;

        /** Returns how paths map to the OS file system
        
        @returns OSPathKind that describes how paths map to the Operating System file system
        */
        virtual SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() = 0;
    };

    #define SLANG_UUID_ISlangFileSystemExt ISlangFileSystemExt::getTypeGuid()

    struct ISlangMutableFileSystem : public ISlangFileSystemExt
    {
        SLANG_COM_INTERFACE(0xa058675c, 0x1d65, 0x452a, { 0x84, 0x58, 0xcc, 0xde, 0xd1, 0x42, 0x71, 0x5 })

        /** Write data to the specified path.

        @param path The path for data to be saved to
        @param data The data to be saved
        @param size The size of the data in bytes
        @returns SLANG_OK if successful (SLANG_E_NOT_IMPLEMENTED if not implemented, or some other error code)
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL saveFile(
            const char* path,
            const void* data,
            size_t size) = 0;

        /** Write data in the form of a blob to the specified path.

        Depending on the implementation writing a blob might be faster/use less memory. It is assumed the 
        blob is *immutable* and that an implementation can reference count it.

        It is not guaranteed loading the same file will return the *same* blob - just a blob with same 
        contents.

        @param path The path for data to be saved to
        @param dataBlob The data to be saved
        @returns SLANG_OK if successful (SLANG_E_NOT_IMPLEMENTED if not implemented, or some other error code)
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL saveFileBlob(
            const char* path,
            ISlangBlob* dataBlob) = 0;

        /** Remove the entry in the path (directory of file). Will only delete an empty directory, if not empty
        will return an error.

        @param path The path to remove 
        @returns SLANG_OK if successful 
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL remove(
            const char* path) = 0;

        /** Create a directory.

        The path to the directory must exist

        @param path To the directory to create. The parent path *must* exist otherwise will return an error.
        @returns SLANG_OK if successful 
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL createDirectory(
            const char* path) = 0;
    };

    #define SLANG_UUID_ISlangMutableFileSystem ISlangMutableFileSystem::getTypeGuid()

    /* Identifies different types of writer target*/
    typedef unsigned int SlangWriterChannelIntegral;
    enum SlangWriterChannel : SlangWriterChannelIntegral
    {
        SLANG_WRITER_CHANNEL_DIAGNOSTIC,
        SLANG_WRITER_CHANNEL_STD_OUTPUT,
        SLANG_WRITER_CHANNEL_STD_ERROR,
        SLANG_WRITER_CHANNEL_COUNT_OF,
    };

    typedef unsigned int SlangWriterModeIntegral;
    enum SlangWriterMode : SlangWriterModeIntegral
    {
        SLANG_WRITER_MODE_TEXT,
        SLANG_WRITER_MODE_BINARY,
    };

    /** A stream typically of text, used for outputting diagnostic as well as other information.
    */
    struct ISlangWriter : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(0xec457f0e, 0x9add, 0x4e6b,{ 0x85, 0x1c, 0xd7, 0xfa, 0x71, 0x6d, 0x15, 0xfd })

            /** Begin an append buffer.
            NOTE! Only one append buffer can be active at any time.
            @param maxNumChars The maximum of chars that will be appended
            @returns The start of the buffer for appending to. */    
        virtual SLANG_NO_THROW char* SLANG_MCALL beginAppendBuffer(size_t maxNumChars) = 0;
            /** Ends the append buffer, and is equivalent to a write of the append buffer.
            NOTE! That an endAppendBuffer is not necessary if there are no characters to write.
            @param buffer is the start of the data to append and must be identical to last value returned from beginAppendBuffer
            @param numChars must be a value less than or equal to what was returned from last call to beginAppendBuffer
            @returns Result, will be SLANG_OK on success */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL endAppendBuffer(char* buffer, size_t numChars) = 0;
            /** Write text to the writer
            @param chars The characters to write out
            @param numChars The amount of characters
            @returns SLANG_OK on success */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL write(const char* chars, size_t numChars) = 0;
            /** Flushes any content to the output */
        virtual SLANG_NO_THROW void SLANG_MCALL flush() = 0;
            /** Determines if the writer stream is to the console, and can be used to alter the output 
            @returns Returns true if is a console writer */
        virtual SLANG_NO_THROW SlangBool SLANG_MCALL isConsole() = 0;
            /** Set the mode for the writer to use
            @param mode The mode to use
            @returns SLANG_OK on success */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL setMode(SlangWriterMode mode) = 0;
    };
    
    #define SLANG_UUID_ISlangWriter ISlangWriter::getTypeGuid()

    struct ISlangProfiler : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(0x197772c7, 0x0155, 0x4b91, { 0x84, 0xe8, 0x66, 0x68, 0xba, 0xff, 0x06, 0x19 })
        virtual SLANG_NO_THROW size_t SLANG_MCALL getEntryCount() = 0;
        virtual SLANG_NO_THROW const char* SLANG_MCALL getEntryName(uint32_t index) = 0;
        virtual SLANG_NO_THROW long SLANG_MCALL getEntryTimeMS(uint32_t index) = 0;
        virtual SLANG_NO_THROW uint32_t SLANG_MCALL getEntryInvocationTimes(uint32_t index) = 0;
    };
    #define SLANG_UUID_ISlangProfiler ISlangProfiler::getTypeGuid()

    namespace slang {
    struct IGlobalSession;
    struct ICompileRequest;

    } // namespace slang

    /*!
    @brief An instance of the Slang library.
    */
    typedef slang::IGlobalSession SlangSession;
    

    typedef struct SlangProgramLayout SlangProgramLayout;

    /*!
    @brief A request for one or more compilation actions to be performed.
    */
    typedef struct slang::ICompileRequest SlangCompileRequest;


    /*!
    @brief Initialize an instance of the Slang library.
    */
    SLANG_API SlangSession* spCreateSession(const char* deprecated = 0);

    /*!
    @brief Clean up after an instance of the Slang library.
    */
    SLANG_API void spDestroySession(
        SlangSession*   session);

    /** @see slang::IGlobalSession::setSharedLibraryLoader
    */
    SLANG_API void spSessionSetSharedLibraryLoader(
        SlangSession*               session,
        ISlangSharedLibraryLoader*  loader);

    /** @see slang::IGlobalSession::getSharedLibraryLoader
    */
    SLANG_API ISlangSharedLibraryLoader* spSessionGetSharedLibraryLoader(
        SlangSession*   session);

    /** @see slang::IGlobalSession::checkCompileTargetSupport
    */
    SLANG_API SlangResult spSessionCheckCompileTargetSupport(
        SlangSession*       session,
        SlangCompileTarget  target);

    /** @see slang::IGlobalSession::checkPassThroughSupport
    */
    SLANG_API SlangResult spSessionCheckPassThroughSupport(
        SlangSession*       session,
        SlangPassThrough    passThrough
    );

    /** @see slang::IGlobalSession::addBuiltins
    */
    SLANG_API void spAddBuiltins(
        SlangSession*   session,
        char const*     sourcePath,
        char const*     sourceString);

        /*!
    @brief Callback type used for diagnostic output. 
    */
    typedef void(*SlangDiagnosticCallback)(
        char const* message,
        void*       userData);

    /*!
    @brief Get the build version 'tag' string. The string is the same as
    produced via `git describe --tags --match v*` for the project. If such a
    version could not be determined at build time then the contents will be
    0.0.0-unknown. Any string can be set by passing
    -DSLANG_VERSION_FULL=whatever during the cmake invocation.

    This function will return exactly the same result as the method
    getBuildTagString on IGlobalSession.

    An advantage of using this function over the method is that doing so does
    not require the creation of a session, which can be a fairly costly
    operation.

    @return The build tag string
    */
    SLANG_API const char* spGetBuildTagString();

    /* @see slang::IGlobalSession::createCompileRequest
    */
    SLANG_API SlangCompileRequest* spCreateCompileRequest(
        SlangSession* session);

    /*!
    @brief Destroy a compile request.
    Note a request is a COM object and can be destroyed via 'Release'.
    */
    SLANG_API void spDestroyCompileRequest(
        SlangCompileRequest*    request);

    /*! @see slang::ICompileRequest::setFileSystem */
    SLANG_API void spSetFileSystem(
        SlangCompileRequest*    request,
        ISlangFileSystem*       fileSystem);

    /*! @see slang::ICompileRequest::setCompileFlags */
    SLANG_API void spSetCompileFlags(
        SlangCompileRequest*    request,
        SlangCompileFlags       flags);

    /*! @see slang::ICompileRequest::getCompileFlags */
    SLANG_API SlangCompileFlags spGetCompileFlags(
        SlangCompileRequest*    request);

    /*! @see slang::ICompileRequest::setDumpIntermediates */
    SLANG_API void spSetDumpIntermediates(
        SlangCompileRequest*    request,
        int                     enable);

    /*! @see slang::ICompileRequest::setDumpIntermediatePrefix */
    SLANG_API void spSetDumpIntermediatePrefix(
        SlangCompileRequest*    request,
        const char* prefix);

    /*! DEPRECATED: use `spSetTargetLineDirectiveMode` instead.
        @see slang::ICompileRequest::setLineDirectiveMode */
    SLANG_API void spSetLineDirectiveMode(
        SlangCompileRequest*    request,
        SlangLineDirectiveMode  mode);
        
    /*! @see slang::ICompileRequest::setTargetLineDirectiveMode */
    SLANG_API void spSetTargetLineDirectiveMode(
        SlangCompileRequest*    request,
        int targetIndex,
        SlangLineDirectiveMode  mode);

    /*! @see slang::ICompileRequest::setTargetLineDirectiveMode */
    SLANG_API void spSetTargetForceGLSLScalarBufferLayout(
        SlangCompileRequest*    request,
        int targetIndex,
        bool forceScalarLayout);

    /*! @see slang::ICompileRequest::setTargetUseMinimumSlangOptimization */
    SLANG_API void spSetTargetUseMinimumSlangOptimization(
        slang::ICompileRequest* request,
        int targetIndex,
        bool val);

    /*! @see slang::ICompileRequest::setIngoreCapabilityCheck */
    SLANG_API void spSetIgnoreCapabilityCheck(
        slang::ICompileRequest* request,
        bool val);

    /*! @see slang::ICompileRequest::setCodeGenTarget */
    SLANG_API void spSetCodeGenTarget(
        SlangCompileRequest*    request,
        SlangCompileTarget target);

    /*! @see slang::ICompileRequest::addCodeGenTarget */
    SLANG_API int spAddCodeGenTarget(
        SlangCompileRequest*    request,
        SlangCompileTarget      target);

    /*! @see slang::ICompileRequest::setTargetProfile */
    SLANG_API void spSetTargetProfile(
        SlangCompileRequest*    request,
        int                     targetIndex,
        SlangProfileID          profile);

    /*! @see slang::ICompileRequest::setTargetFlags */
    SLANG_API void spSetTargetFlags(
        SlangCompileRequest*    request,
        int                     targetIndex,
        SlangTargetFlags        flags);



    /*! @see slang::ICompileRequest::setTargetFloatingPointMode */
    SLANG_API void spSetTargetFloatingPointMode(
        SlangCompileRequest*    request,
        int                     targetIndex,
        SlangFloatingPointMode  mode);

    /*! @see slang::ICompileRequest::addTargetCapability */
    SLANG_API void spAddTargetCapability(
        slang::ICompileRequest* request,
        int                     targetIndex,
        SlangCapabilityID       capability);

    /* DEPRECATED: use `spSetMatrixLayoutMode` instead. */
    SLANG_API void spSetTargetMatrixLayoutMode(
        SlangCompileRequest*    request,
        int                     targetIndex,
        SlangMatrixLayoutMode   mode);

    /*! @see slang::ICompileRequest::setMatrixLayoutMode */
    SLANG_API void spSetMatrixLayoutMode(
        SlangCompileRequest*    request,
        SlangMatrixLayoutMode   mode);

    /*! @see slang::ICompileRequest::setDebugInfoLevel */
    SLANG_API void spSetDebugInfoLevel(
        SlangCompileRequest*    request,
        SlangDebugInfoLevel     level);

    /*! @see slang::ICompileRequest::setDebugInfoFormat */
    SLANG_API void spSetDebugInfoFormat(
        SlangCompileRequest*    request,
        SlangDebugInfoFormat        format);

    /*! @see slang::ICompileRequest::setOptimizationLevel */
    SLANG_API void spSetOptimizationLevel(
        SlangCompileRequest*    request,
        SlangOptimizationLevel  level);


    
    /*! @see slang::ICompileRequest::setOutputContainerFormat */
    SLANG_API void spSetOutputContainerFormat(
        SlangCompileRequest*    request,
        SlangContainerFormat    format);

    /*! @see slang::ICompileRequest::setPassThrough */
    SLANG_API void spSetPassThrough(
        SlangCompileRequest*    request,
        SlangPassThrough        passThrough);

     /*! @see slang::ICompileRequest::setDiagnosticCallback */
    SLANG_API void spSetDiagnosticCallback(
        SlangCompileRequest*    request,
        SlangDiagnosticCallback callback,
        void const*             userData);

    /*! @see slang::ICompileRequest::setWriter */
    SLANG_API void spSetWriter(
        SlangCompileRequest*    request,
        SlangWriterChannel      channel, 
        ISlangWriter*           writer);

    /*! @see slang::ICompileRequest::getWriter */
    SLANG_API ISlangWriter* spGetWriter(
        SlangCompileRequest*    request,
        SlangWriterChannel      channel);

    /*! @see slang::ICompileRequest::addSearchPath */
    SLANG_API void spAddSearchPath(
        SlangCompileRequest*    request,
        const char*             searchDir);

   /*! @see slang::ICompileRequest::addPreprocessorDefine */
    SLANG_API void spAddPreprocessorDefine(
        SlangCompileRequest*    request,
        const char*             key,
        const char*             value);

    /*! @see slang::ICompileRequest::processCommandLineArguments */
    SLANG_API SlangResult spProcessCommandLineArguments(
        SlangCompileRequest*    request,
        char const* const*      args,
        int                     argCount);

    /*! @see slang::ICompileRequest::addTranslationUnit */
    SLANG_API int spAddTranslationUnit(
        SlangCompileRequest*    request,
        SlangSourceLanguage     language,
        char const*             name);

    
    /*! @see slang::ICompileRequest::setDefaultModuleName */
    SLANG_API void spSetDefaultModuleName(
        SlangCompileRequest*    request,
        const char* defaultModuleName);

    /*! @see slang::ICompileRequest::addPreprocessorDefine */
    SLANG_API void spTranslationUnit_addPreprocessorDefine(
        SlangCompileRequest*    request,
        int                     translationUnitIndex,
        const char*             key,
        const char*             value);


    /*! @see slang::ICompileRequest::addTranslationUnitSourceFile */
    SLANG_API void spAddTranslationUnitSourceFile(
        SlangCompileRequest*    request,
        int                     translationUnitIndex,
        char const*             path);

    /*! @see slang::ICompileRequest::addTranslationUnitSourceString */
    SLANG_API void spAddTranslationUnitSourceString(
        SlangCompileRequest*    request,
        int                     translationUnitIndex,
        char const*             path,
        char const*             source);


    /*! @see slang::ICompileRequest::addLibraryReference */
    SLANG_API SlangResult spAddLibraryReference(
        SlangCompileRequest*    request,
        const char* basePath,
        const void* libData,
        size_t libDataSize);

    /*! @see slang::ICompileRequest::addTranslationUnitSourceStringSpan */
    SLANG_API void spAddTranslationUnitSourceStringSpan(
        SlangCompileRequest*    request,
        int                     translationUnitIndex,
        char const*             path,
        char const*             sourceBegin,
        char const*             sourceEnd);

    /*! @see slang::ICompileRequest::addTranslationUnitSourceBlob */
    SLANG_API void spAddTranslationUnitSourceBlob(
        SlangCompileRequest*    request,
        int                     translationUnitIndex,
        char const*             path,
        ISlangBlob*             sourceBlob);

    /*! @see slang::IGlobalSession::findProfile */
    SLANG_API SlangProfileID spFindProfile(
        SlangSession*   session,
        char const*     name);

    /*! @see slang::IGlobalSession::findCapability */
    SLANG_API SlangCapabilityID spFindCapability(
        SlangSession*   session,
        char const*     name);

    /*! @see slang::ICompileRequest::addEntryPoint */
    SLANG_API int spAddEntryPoint(
        SlangCompileRequest*    request,
        int                     translationUnitIndex,
        char const*             name,
        SlangStage              stage);

    /*! @see slang::ICompileRequest::addEntryPointEx */
    SLANG_API int spAddEntryPointEx(
        SlangCompileRequest*    request,
        int                     translationUnitIndex,
        char const*             name,
        SlangStage              stage,
        int                     genericArgCount,
        char const**            genericArgs);

    /*! @see slang::ICompileRequest::setGlobalGenericArgs */
    SLANG_API SlangResult spSetGlobalGenericArgs(
        SlangCompileRequest*    request,
        int                     genericArgCount,
        char const**            genericArgs);

    /*! @see slang::ICompileRequest::setTypeNameForGlobalExistentialTypeParam */
    SLANG_API SlangResult spSetTypeNameForGlobalExistentialTypeParam(
        SlangCompileRequest*    request,
        int                     slotIndex,
        char const*             typeName);

    /*! @see slang::ICompileRequest::setTypeNameForEntryPointExistentialTypeParam */
    SLANG_API SlangResult spSetTypeNameForEntryPointExistentialTypeParam(
        SlangCompileRequest*    request,
        int                     entryPointIndex,
        int                     slotIndex,
        char const*             typeName);

    /*! @see slang::ICompileRequest::compile */
    SLANG_API SlangResult spCompile(
        SlangCompileRequest*    request);


    /*! @see slang::ICompileRequest::getDiagnosticOutput */
    SLANG_API char const* spGetDiagnosticOutput(
        SlangCompileRequest*    request);

    /*! @see slang::ICompileRequest::getDiagnosticOutputBlob */
    SLANG_API SlangResult spGetDiagnosticOutputBlob(
        SlangCompileRequest*    request,
        ISlangBlob**            outBlob);


    /*! @see slang::ICompileRequest::getDependencyFileCount */
    SLANG_API int
    spGetDependencyFileCount(
        SlangCompileRequest*    request);

    /*! @see slang::ICompileRequest::getDependencyFilePath */
    SLANG_API char const*
    spGetDependencyFilePath(
        SlangCompileRequest*    request,
        int                     index);

    /*! @see slang::ICompileRequest::getTranslationUnitCount */
    SLANG_API int
    spGetTranslationUnitCount(
        SlangCompileRequest*    request);

    /*! @see slang::ICompileRequest::getEntryPointSource */
    SLANG_API char const* spGetEntryPointSource(
        SlangCompileRequest*    request,
        int                     entryPointIndex);

    /*! @see slang::ICompileRequest::getEntryPointCode */
    SLANG_API void const* spGetEntryPointCode(
        SlangCompileRequest*    request,
        int                     entryPointIndex,
        size_t*                 outSize);

    /*! @see slang::ICompileRequest::getEntryPointCodeBlob */
    SLANG_API SlangResult spGetEntryPointCodeBlob(
        SlangCompileRequest*    request,
        int                     entryPointIndex,
        int                     targetIndex,
        ISlangBlob**            outBlob);

    /*! @see slang::ICompileRequest::getEntryPointHostCallable */
    SLANG_API SlangResult spGetEntryPointHostCallable(
        SlangCompileRequest*    request,
        int                     entryPointIndex,
        int                     targetIndex,
        ISlangSharedLibrary**   outSharedLibrary);

    /*! @see slang::ICompileRequest::getTargetCodeBlob */
    SLANG_API SlangResult spGetTargetCodeBlob(
        SlangCompileRequest*    request,
        int                     targetIndex,
        ISlangBlob**            outBlob);

    /*! @see slang::ICompileRequest::getTargetHostCallable */
    SLANG_API SlangResult spGetTargetHostCallable(
        SlangCompileRequest*    request,
        int                     targetIndex,
        ISlangSharedLibrary**   outSharedLibrary);

    /*! @see slang::ICompileRequest::getCompileRequestCode */
    SLANG_API void const* spGetCompileRequestCode(
        SlangCompileRequest*    request,
        size_t*                 outSize);

    /*! @see slang::ICompileRequest::getContainerCode */
    SLANG_API SlangResult spGetContainerCode(
        SlangCompileRequest*    request,
        ISlangBlob**            outBlob);

    /*! @see slang::ICompileRequest::loadRepro */
    SLANG_API SlangResult spLoadRepro(
        SlangCompileRequest* request,
        ISlangFileSystem* fileSystem,
        const void* data,
        size_t size);

    /*! @see slang::ICompileRequest::saveRepro */
    SLANG_API SlangResult spSaveRepro(
        SlangCompileRequest* request,
        ISlangBlob** outBlob
    );

    /*! @see slang::ICompileRequest::enableReproCapture */
    SLANG_API SlangResult spEnableReproCapture(
        SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::getCompileTimeProfile */
    SLANG_API SlangResult spGetCompileTimeProfile(
        SlangCompileRequest* request,
        ISlangProfiler** compileTimeProfile,
        bool shouldClear);


    /** Extract contents of a repro.

    Writes the contained files and manifest with their 'unique' names into fileSystem. For more details read the
    docs/repro.md documentation. 

    @param session          The slang session
    @param reproData        Holds the repro data
    @param reproDataSize    The size of the repro data
    @param fileSystem       File system that the contents of the repro will be written to
    @returns                A `SlangResult` to indicate success or failure.
    */
    SLANG_API SlangResult spExtractRepro(
        SlangSession* session,
        const void* reproData,
        size_t reproDataSize,
        ISlangMutableFileSystem* fileSystem);

    /* Turns a repro into a file system.

    Makes the contents of the repro available as a file system - that is able to access the files with the same
    paths as were used on the original repro file system. 

    @param session          The slang session
    @param reproData        The repro data
    @param reproDataSize    The size of the repro data
    @param replaceFileSystem  Will attempt to load by unique names from this file system before using contents of the repro. Optional.
    @param outFileSystem    The file system that can be used to access contents
    @returns                A `SlangResult` to indicate success or failure.
    */
    SLANG_API SlangResult spLoadReproAsFileSystem(
        SlangSession* session,
        const void* reproData,
        size_t reproDataSize,
        ISlangFileSystem* replaceFileSystem,
        ISlangFileSystemExt** outFileSystem);

    /*! @see slang::ICompileRequest::overrideDiagnosticSeverity */
    SLANG_API void spOverrideDiagnosticSeverity(
        SlangCompileRequest* request,
        SlangInt messageID,
        SlangSeverity overrideSeverity);

    /*! @see slang::ICompileRequest::getDiagnosticFlags */
    SLANG_API SlangDiagnosticFlags spGetDiagnosticFlags(SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::setDiagnosticFlags */
    SLANG_API void spSetDiagnosticFlags(SlangCompileRequest* request, SlangDiagnosticFlags flags);

    /*
    Forward declarations of types used in the reflection interface;
    */

    typedef struct SlangProgramLayout SlangProgramLayout;
    typedef struct SlangEntryPoint SlangEntryPoint;
    typedef struct SlangEntryPointLayout SlangEntryPointLayout;

    typedef struct SlangReflectionModifier          SlangReflectionModifier;
    typedef struct SlangReflectionType              SlangReflectionType;
    typedef struct SlangReflectionTypeLayout        SlangReflectionTypeLayout;
    typedef struct SlangReflectionVariable          SlangReflectionVariable;
    typedef struct SlangReflectionVariableLayout    SlangReflectionVariableLayout;
    typedef struct SlangReflectionTypeParameter     SlangReflectionTypeParameter;
    typedef struct SlangReflectionUserAttribute     SlangReflectionUserAttribute;
    typedef struct SlangReflectionFunction          SlangReflectionFunction;

    /*
    Type aliases to maintain backward compatibility.
    */
    typedef SlangProgramLayout SlangReflection;
    typedef SlangEntryPointLayout SlangReflectionEntryPoint;

    // get reflection data from a compilation request
    SLANG_API SlangReflection* spGetReflection(
        SlangCompileRequest*    request);

    // type reflection

    typedef unsigned int SlangTypeKindIntegral;
    enum SlangTypeKind : SlangTypeKindIntegral
    {
        SLANG_TYPE_KIND_NONE,
        SLANG_TYPE_KIND_STRUCT,
        SLANG_TYPE_KIND_ARRAY,
        SLANG_TYPE_KIND_MATRIX,
        SLANG_TYPE_KIND_VECTOR,
        SLANG_TYPE_KIND_SCALAR,
        SLANG_TYPE_KIND_CONSTANT_BUFFER,
        SLANG_TYPE_KIND_RESOURCE,
        SLANG_TYPE_KIND_SAMPLER_STATE,
        SLANG_TYPE_KIND_TEXTURE_BUFFER,
        SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER,
        SLANG_TYPE_KIND_PARAMETER_BLOCK,
        SLANG_TYPE_KIND_GENERIC_TYPE_PARAMETER,
        SLANG_TYPE_KIND_INTERFACE,
        SLANG_TYPE_KIND_OUTPUT_STREAM,
        SLANG_TYPE_KIND_MESH_OUTPUT,
        SLANG_TYPE_KIND_SPECIALIZED,
        SLANG_TYPE_KIND_FEEDBACK,
        SLANG_TYPE_KIND_POINTER,
        SLANG_TYPE_KIND_COUNT,
    };

    typedef unsigned int SlangScalarTypeIntegral;
    enum SlangScalarType : SlangScalarTypeIntegral
    {
        SLANG_SCALAR_TYPE_NONE,
        SLANG_SCALAR_TYPE_VOID,
        SLANG_SCALAR_TYPE_BOOL,
        SLANG_SCALAR_TYPE_INT32,
        SLANG_SCALAR_TYPE_UINT32,
        SLANG_SCALAR_TYPE_INT64,
        SLANG_SCALAR_TYPE_UINT64,
        SLANG_SCALAR_TYPE_FLOAT16,
        SLANG_SCALAR_TYPE_FLOAT32,
        SLANG_SCALAR_TYPE_FLOAT64,
        SLANG_SCALAR_TYPE_INT8,
        SLANG_SCALAR_TYPE_UINT8,
        SLANG_SCALAR_TYPE_INT16,
        SLANG_SCALAR_TYPE_UINT16,
        SLANG_SCALAR_TYPE_INTPTR,
        SLANG_SCALAR_TYPE_UINTPTR
    };

#ifndef SLANG_RESOURCE_SHAPE
#    define SLANG_RESOURCE_SHAPE
    typedef unsigned int SlangResourceShapeIntegral;
    enum SlangResourceShape : SlangResourceShapeIntegral
    {
        SLANG_RESOURCE_BASE_SHAPE_MASK      = 0x0F,

        SLANG_RESOURCE_NONE                 = 0x00,

        SLANG_TEXTURE_1D                    = 0x01,
        SLANG_TEXTURE_2D                    = 0x02,
        SLANG_TEXTURE_3D                    = 0x03,
        SLANG_TEXTURE_CUBE                  = 0x04,
        SLANG_TEXTURE_BUFFER                = 0x05,

        SLANG_STRUCTURED_BUFFER             = 0x06,
        SLANG_BYTE_ADDRESS_BUFFER           = 0x07,
        SLANG_RESOURCE_UNKNOWN              = 0x08,
        SLANG_ACCELERATION_STRUCTURE        = 0x09,
        SLANG_TEXTURE_SUBPASS               = 0x0A,

        SLANG_RESOURCE_EXT_SHAPE_MASK       = 0xF0,

        SLANG_TEXTURE_FEEDBACK_FLAG         = 0x10,
        SLANG_TEXTURE_SHADOW_FLAG           = 0x20,
        SLANG_TEXTURE_ARRAY_FLAG            = 0x40,
        SLANG_TEXTURE_MULTISAMPLE_FLAG      = 0x80,

        SLANG_TEXTURE_1D_ARRAY              = SLANG_TEXTURE_1D   | SLANG_TEXTURE_ARRAY_FLAG,
        SLANG_TEXTURE_2D_ARRAY              = SLANG_TEXTURE_2D   | SLANG_TEXTURE_ARRAY_FLAG,
        SLANG_TEXTURE_CUBE_ARRAY            = SLANG_TEXTURE_CUBE | SLANG_TEXTURE_ARRAY_FLAG,

        SLANG_TEXTURE_2D_MULTISAMPLE        = SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG,
        SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY  = SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG | SLANG_TEXTURE_ARRAY_FLAG,
        SLANG_TEXTURE_SUBPASS_MULTISAMPLE   = SLANG_TEXTURE_SUBPASS | SLANG_TEXTURE_MULTISAMPLE_FLAG,
    };
#endif
    typedef unsigned int SlangResourceAccessIntegral;
    enum SlangResourceAccess : SlangResourceAccessIntegral
    {
        SLANG_RESOURCE_ACCESS_NONE,
        SLANG_RESOURCE_ACCESS_READ,
        SLANG_RESOURCE_ACCESS_READ_WRITE,
        SLANG_RESOURCE_ACCESS_RASTER_ORDERED,
        SLANG_RESOURCE_ACCESS_APPEND,
        SLANG_RESOURCE_ACCESS_CONSUME,
        SLANG_RESOURCE_ACCESS_WRITE,
        SLANG_RESOURCE_ACCESS_FEEDBACK,
        SLANG_RESOURCE_ACCESS_UNKNOWN = 0x7FFFFFFF,
    };

    typedef unsigned int SlangParameterCategoryIntegral;
    enum SlangParameterCategory : SlangParameterCategoryIntegral
    {
        SLANG_PARAMETER_CATEGORY_NONE,
        SLANG_PARAMETER_CATEGORY_MIXED,
        SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
        SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE,
        SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS,
        SLANG_PARAMETER_CATEGORY_VARYING_INPUT,
        SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT,
        SLANG_PARAMETER_CATEGORY_SAMPLER_STATE,
        SLANG_PARAMETER_CATEGORY_UNIFORM,
        SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT,
        SLANG_PARAMETER_CATEGORY_SPECIALIZATION_CONSTANT,
        SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER,

        // HLSL register `space`, Vulkan GLSL `set`
        SLANG_PARAMETER_CATEGORY_REGISTER_SPACE,

        // TODO: Ellie, Both APIs treat mesh outputs as more or less varying output,
        // Does it deserve to be represented here??

        // A parameter whose type is to be specialized by a global generic type argument
        SLANG_PARAMETER_CATEGORY_GENERIC,

        SLANG_PARAMETER_CATEGORY_RAY_PAYLOAD,
        SLANG_PARAMETER_CATEGORY_HIT_ATTRIBUTES,
        SLANG_PARAMETER_CATEGORY_CALLABLE_PAYLOAD,
        SLANG_PARAMETER_CATEGORY_SHADER_RECORD,

        // An existential type parameter represents a "hole" that
        // needs to be filled with a concrete type to enable
        // generation of specialized code.
        //
        // Consider this example:
        //
        //      struct MyParams
        //      {
        //          IMaterial material;
        //          ILight lights[3];
        //      };
        //
        // This `MyParams` type introduces two existential type parameters:
        // one for `material` and one for `lights`. Even though `lights`
        // is an array, it only introduces one type parameter, because
        // we need to hae a *single* concrete type for all the array
        // elements to be able to generate specialized code.
        //
        SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM,

        // An existential object parameter represents a value
        // that needs to be passed in to provide data for some
        // interface-type shader paameter.
        //
        // Consider this example:
        //
        //      struct MyParams
        //      {
        //          IMaterial material;
        //          ILight lights[3];
        //      };
        //
        // This `MyParams` type introduces four existential object parameters:
        // one for `material` and three for `lights` (one for each array
        // element). This is consistent with the number of interface-type
        // "objects" that are being passed through to the shader.
        //
        SLANG_PARAMETER_CATEGORY_EXISTENTIAL_OBJECT_PARAM,

        // The register space offset for the sub-elements that occupies register spaces.
        SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE,

        // The input_attachment_index subpass occupancy tracker
        SLANG_PARAMETER_CATEGORY_SUBPASS,

        // Metal resource binding points.
        SLANG_PARAMETER_CATEGORY_METAL_ARGUMENT_BUFFER_ELEMENT,

        // Metal [[attribute]] inputs.
        SLANG_PARAMETER_CATEGORY_METAL_ATTRIBUTE,

        // Metal [[payload]] inputs
        SLANG_PARAMETER_CATEGORY_METAL_PAYLOAD,

        //
        SLANG_PARAMETER_CATEGORY_COUNT,

        // Aliases for Metal-specific categories.
        SLANG_PARAMETER_CATEGORY_METAL_BUFFER = SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
        SLANG_PARAMETER_CATEGORY_METAL_TEXTURE = SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE,
        SLANG_PARAMETER_CATEGORY_METAL_SAMPLER = SLANG_PARAMETER_CATEGORY_SAMPLER_STATE,

        // DEPRECATED:
        SLANG_PARAMETER_CATEGORY_VERTEX_INPUT = SLANG_PARAMETER_CATEGORY_VARYING_INPUT,
        SLANG_PARAMETER_CATEGORY_FRAGMENT_OUTPUT = SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT,
        SLANG_PARAMETER_CATEGORY_COUNT_V1 = SLANG_PARAMETER_CATEGORY_SUBPASS,
    };

    /** Types of API-managed bindings that a parameter might use.
    
    `SlangBindingType` represents the distinct types of binding ranges that might be
    understood by an underlying graphics API or cross-API abstraction layer.
    Several of the enumeration cases here correspond to cases of `VkDescriptorType`
    defined by the Vulkan API. Note however that the values of this enumeration
    are not the same as those of any particular API.

    The `SlangBindingType` enumeration is distinct from `SlangParameterCategory`
    because `SlangParameterCategory` differentiates the types of parameters for
    the purposes of layout, where the layout rules of some targets will treat
    parameters of different types as occupying the same binding space for layout
    (e.g., in SPIR-V both a `Texture2D` and `SamplerState` use the same space of
    `binding` indices, and are not allowed to overlap), while those same types
    map to different types of bindingsin the API (e.g., both textures and samplers
    use different `VkDescriptorType` values).

    When you want to answer "what register/binding did this parameter use?" you
    should use `SlangParameterCategory`.

    When you wnat to answer "what type of descriptor range should this parameter use?"
    you should use `SlangBindingType`.
    */
    typedef SlangUInt32 SlangBindingTypeIntegral;
    enum SlangBindingType : SlangBindingTypeIntegral
    {
        SLANG_BINDING_TYPE_UNKNOWN = 0,

        SLANG_BINDING_TYPE_SAMPLER,
        SLANG_BINDING_TYPE_TEXTURE,
        SLANG_BINDING_TYPE_CONSTANT_BUFFER,
        SLANG_BINDING_TYPE_PARAMETER_BLOCK,
        SLANG_BINDING_TYPE_TYPED_BUFFER,
        SLANG_BINDING_TYPE_RAW_BUFFER,
        SLANG_BINDING_TYPE_COMBINED_TEXTURE_SAMPLER,
        SLANG_BINDING_TYPE_INPUT_RENDER_TARGET,
        SLANG_BINDING_TYPE_INLINE_UNIFORM_DATA,
        SLANG_BINDING_TYPE_RAY_TRACING_ACCELERATION_STRUCTURE,

        SLANG_BINDING_TYPE_VARYING_INPUT,
        SLANG_BINDING_TYPE_VARYING_OUTPUT,

        SLANG_BINDING_TYPE_EXISTENTIAL_VALUE,
        SLANG_BINDING_TYPE_PUSH_CONSTANT,

        SLANG_BINDING_TYPE_MUTABLE_FLAG = 0x100,

        SLANG_BINDING_TYPE_MUTABLE_TETURE = SLANG_BINDING_TYPE_TEXTURE | SLANG_BINDING_TYPE_MUTABLE_FLAG,
        SLANG_BINDING_TYPE_MUTABLE_TYPED_BUFFER = SLANG_BINDING_TYPE_TYPED_BUFFER | SLANG_BINDING_TYPE_MUTABLE_FLAG,
        SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER = SLANG_BINDING_TYPE_RAW_BUFFER | SLANG_BINDING_TYPE_MUTABLE_FLAG,

        SLANG_BINDING_TYPE_BASE_MASK = 0x00FF,
        SLANG_BINDING_TYPE_EXT_MASK  = 0xFF00,
    };

    typedef SlangUInt32 SlangLayoutRulesIntegral;
    enum SlangLayoutRules : SlangLayoutRulesIntegral
    {
        SLANG_LAYOUT_RULES_DEFAULT,
    };

    typedef SlangUInt32 SlangModifierIDIntegral;
    enum SlangModifierID : SlangModifierIDIntegral
    {
        SLANG_MODIFIER_SHARED,
        SLANG_MODIFIER_NO_DIFF,
        SLANG_MODIFIER_STATIC,
        SLANG_MODIFIER_CONST,
        SLANG_MODIFIER_EXPORT,
        SLANG_MODIFIER_EXTERN,
        SLANG_MODIFIER_DIFFERENTIABLE,
    };

    // User Attribute
    SLANG_API char const* spReflectionUserAttribute_GetName(SlangReflectionUserAttribute* attrib);
    SLANG_API unsigned int spReflectionUserAttribute_GetArgumentCount(SlangReflectionUserAttribute* attrib);
    SLANG_API SlangReflectionType* spReflectionUserAttribute_GetArgumentType(SlangReflectionUserAttribute* attrib, unsigned int index);
    SLANG_API SlangResult spReflectionUserAttribute_GetArgumentValueInt(SlangReflectionUserAttribute* attrib, unsigned int index, int * rs);
    SLANG_API SlangResult spReflectionUserAttribute_GetArgumentValueFloat(SlangReflectionUserAttribute* attrib, unsigned int index, float * rs);

    /** Returns the string-typed value of a user attribute argument
        The string returned is not null-terminated. The length of the string is returned via `outSize`.
        If index of out of range, or if the specified argument is not a string, the function will return nullptr.
    */
    SLANG_API const char* spReflectionUserAttribute_GetArgumentValueString(SlangReflectionUserAttribute* attrib, unsigned int index, size_t * outSize);

    // Type Reflection

    SLANG_API SlangTypeKind spReflectionType_GetKind(SlangReflectionType* type);
    SLANG_API unsigned int spReflectionType_GetUserAttributeCount(SlangReflectionType* type);
    SLANG_API SlangReflectionUserAttribute* spReflectionType_GetUserAttribute(SlangReflectionType* type, unsigned int index);
    SLANG_API SlangReflectionUserAttribute* spReflectionType_FindUserAttributeByName(SlangReflectionType* type, char const* name);

    SLANG_API unsigned int spReflectionType_GetFieldCount(SlangReflectionType* type);
    SLANG_API SlangReflectionVariable* spReflectionType_GetFieldByIndex(SlangReflectionType* type, unsigned index);

        /** Returns the number of elements in the given type.

        This operation is valid for vector and array types. For other types it returns zero.

        When invoked on an unbounded-size array it will return `SLANG_UNBOUNDED_SIZE`,
        which is defined to be `~size_t(0)`.

        If the size of a type cannot be statically computed, perhaps because it depends on
        a generic parameter that has not been bound to a specific value, this function returns zero.
        */
    SLANG_API size_t spReflectionType_GetElementCount(SlangReflectionType* type);

    #define SLANG_UNBOUNDED_SIZE (~size_t(0))

    SLANG_API SlangReflectionType* spReflectionType_GetElementType(SlangReflectionType* type);

    SLANG_API unsigned int spReflectionType_GetRowCount(SlangReflectionType* type);
    SLANG_API unsigned int spReflectionType_GetColumnCount(SlangReflectionType* type);
    SLANG_API SlangScalarType spReflectionType_GetScalarType(SlangReflectionType* type);

    SLANG_API SlangResourceShape spReflectionType_GetResourceShape(SlangReflectionType* type);
    SLANG_API SlangResourceAccess spReflectionType_GetResourceAccess(SlangReflectionType* type);
    SLANG_API SlangReflectionType* spReflectionType_GetResourceResultType(SlangReflectionType* type);

    SLANG_API char const* spReflectionType_GetName(SlangReflectionType* type);
    SLANG_API SlangResult spReflectionType_GetFullName(SlangReflectionType* type, ISlangBlob** outNameBlob);

    // Type Layout Reflection

    SLANG_API SlangReflectionType* spReflectionTypeLayout_GetType(SlangReflectionTypeLayout* type);
    SLANG_API SlangTypeKind spReflectionTypeLayout_getKind(SlangReflectionTypeLayout* type);
    SLANG_API size_t spReflectionTypeLayout_GetSize(SlangReflectionTypeLayout* type, SlangParameterCategory category);
    SLANG_API size_t spReflectionTypeLayout_GetStride(SlangReflectionTypeLayout* type, SlangParameterCategory category);
    SLANG_API int32_t spReflectionTypeLayout_getAlignment(SlangReflectionTypeLayout* type, SlangParameterCategory category);

    SLANG_API uint32_t spReflectionTypeLayout_GetFieldCount(SlangReflectionTypeLayout* type);
    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetFieldByIndex(SlangReflectionTypeLayout* type, unsigned index);

    SLANG_API SlangInt spReflectionTypeLayout_findFieldIndexByName(SlangReflectionTypeLayout* typeLayout, const char* nameBegin, const char* nameEnd);

    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetExplicitCounter(SlangReflectionTypeLayout* typeLayout);

    SLANG_API size_t spReflectionTypeLayout_GetElementStride(SlangReflectionTypeLayout* type, SlangParameterCategory category);
    SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_GetElementTypeLayout(SlangReflectionTypeLayout* type);
    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetElementVarLayout(SlangReflectionTypeLayout* type);
    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_getContainerVarLayout(SlangReflectionTypeLayout* type);

    SLANG_API SlangParameterCategory spReflectionTypeLayout_GetParameterCategory(SlangReflectionTypeLayout* type);

    SLANG_API unsigned spReflectionTypeLayout_GetCategoryCount(SlangReflectionTypeLayout* type);
    SLANG_API SlangParameterCategory spReflectionTypeLayout_GetCategoryByIndex(SlangReflectionTypeLayout* type, unsigned index);

    SLANG_API SlangMatrixLayoutMode spReflectionTypeLayout_GetMatrixLayoutMode(SlangReflectionTypeLayout* type);

    SLANG_API int spReflectionTypeLayout_getGenericParamIndex(SlangReflectionTypeLayout* type);

    SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getPendingDataTypeLayout(SlangReflectionTypeLayout* type);

    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_getSpecializedTypePendingDataVarLayout(SlangReflectionTypeLayout* type);
    SLANG_API SlangInt spReflectionType_getSpecializedTypeArgCount(SlangReflectionType* type);
    SLANG_API SlangReflectionType* spReflectionType_getSpecializedTypeArgType(SlangReflectionType* type, SlangInt index);

    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeCount(SlangReflectionTypeLayout* typeLayout);
    SLANG_API SlangBindingType spReflectionTypeLayout_getBindingRangeType(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_isBindingRangeSpecializable(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeBindingCount(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getBindingRangeLeafTypeLayout(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangReflectionVariable* spReflectionTypeLayout_getBindingRangeLeafVariable(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getFieldBindingRangeOffset(SlangReflectionTypeLayout* typeLayout, SlangInt fieldIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getExplicitCounterBindingRangeOffset(SlangReflectionTypeLayout* inTypeLayout);

    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeDescriptorSetIndex(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeFirstDescriptorRangeIndex(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeDescriptorRangeCount(SlangReflectionTypeLayout* typeLayout, SlangInt index);

    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetCount(SlangReflectionTypeLayout* typeLayout);
    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetSpaceOffset(SlangReflectionTypeLayout* typeLayout, SlangInt setIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeCount(SlangReflectionTypeLayout* typeLayout, SlangInt setIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeIndexOffset(SlangReflectionTypeLayout* typeLayout, SlangInt setIndex, SlangInt rangeIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeDescriptorCount(SlangReflectionTypeLayout* typeLayout, SlangInt setIndex, SlangInt rangeIndex);
    SLANG_API SlangBindingType spReflectionTypeLayout_getDescriptorSetDescriptorRangeType(SlangReflectionTypeLayout* typeLayout, SlangInt setIndex, SlangInt rangeIndex);
    SLANG_API SlangParameterCategory spReflectionTypeLayout_getDescriptorSetDescriptorRangeCategory(SlangReflectionTypeLayout* typeLayout, SlangInt setIndex, SlangInt rangeIndex);

    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeCount(SlangReflectionTypeLayout* typeLayout);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeBindingRangeIndex(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeSpaceOffset(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex);
    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_getSubObjectRangeOffset(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex);

#if 0
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeCount(SlangReflectionTypeLayout* typeLayout);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeObjectCount(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeBindingRangeIndex(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getSubObjectRangeTypeLayout(SlangReflectionTypeLayout* typeLayout, SlangInt index);

    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeCount(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex);
    SLANG_API SlangBindingType spReflectionTypeLayout_getSubObjectRangeDescriptorRangeBindingType(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeBindingCount(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeIndexOffset(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeSpaceOffset(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject);
#endif

    // Variable Reflection

    SLANG_API char const* spReflectionVariable_GetName(SlangReflectionVariable* var);
    SLANG_API SlangReflectionType* spReflectionVariable_GetType(SlangReflectionVariable* var);
    SLANG_API SlangReflectionModifier* spReflectionVariable_FindModifier(SlangReflectionVariable* var, SlangModifierID modifierID);
    SLANG_API unsigned int spReflectionVariable_GetUserAttributeCount(SlangReflectionVariable* var);
    SLANG_API SlangReflectionUserAttribute* spReflectionVariable_GetUserAttribute(SlangReflectionVariable* var, unsigned int index);
    SLANG_API SlangReflectionUserAttribute* spReflectionVariable_FindUserAttributeByName(SlangReflectionVariable* var, SlangSession * globalSession, char const* name);

    // Variable Layout Reflection

    SLANG_API SlangReflectionVariable* spReflectionVariableLayout_GetVariable(SlangReflectionVariableLayout* var);

    SLANG_API SlangReflectionTypeLayout* spReflectionVariableLayout_GetTypeLayout(SlangReflectionVariableLayout* var);

    SLANG_API size_t spReflectionVariableLayout_GetOffset(SlangReflectionVariableLayout* var, SlangParameterCategory category);
    SLANG_API size_t spReflectionVariableLayout_GetSpace(SlangReflectionVariableLayout* var, SlangParameterCategory category);

    SLANG_API char const* spReflectionVariableLayout_GetSemanticName(SlangReflectionVariableLayout* var);
    SLANG_API size_t spReflectionVariableLayout_GetSemanticIndex(SlangReflectionVariableLayout* var);


    // Function Reflection

    SLANG_API char const* spReflectionFunction_GetName(SlangReflectionFunction* func);
    SLANG_API unsigned int spReflectionFunction_GetUserAttributeCount(SlangReflectionFunction* func);
    SLANG_API SlangReflectionUserAttribute* spReflectionFunction_GetUserAttribute(SlangReflectionFunction* func, unsigned int index);
    SLANG_API SlangReflectionUserAttribute* spReflectionFunction_FindUserAttributeByName(SlangReflectionFunction* func, SlangSession* globalSession, char const* name);
    SLANG_API unsigned int spReflectionFunction_GetParameterCount(SlangReflectionFunction* func);
    SLANG_API SlangReflectionVariable* spReflectionFunction_GetParameter(SlangReflectionFunction* func, unsigned index);
    SLANG_API SlangReflectionType* spReflectionFunction_GetResultType(SlangReflectionFunction* func);

    /** Get the stage that a variable belongs to (if any).

    A variable "belongs" to a specific stage when it is a varying input/output
    parameter either defined as part of the parameter list for an entry
    point *or* at the global scope of a stage-specific GLSL code file (e.g.,
    an `in` parameter in a GLSL `.vs` file belongs to the vertex stage).
    */
    SLANG_API SlangStage spReflectionVariableLayout_getStage(
        SlangReflectionVariableLayout* var);


    SLANG_API SlangReflectionVariableLayout* spReflectionVariableLayout_getPendingDataLayout(SlangReflectionVariableLayout* var);

    // Shader Parameter Reflection

    typedef SlangReflectionVariableLayout SlangReflectionParameter;

    SLANG_API unsigned spReflectionParameter_GetBindingIndex(SlangReflectionParameter* parameter);
    SLANG_API unsigned spReflectionParameter_GetBindingSpace(SlangReflectionParameter* parameter);

    SLANG_API SlangResult spIsParameterLocationUsed(
        SlangCompileRequest* request,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        SlangParameterCategory category, // is this a `t` register? `s` register?
        SlangUInt spaceIndex,      // `space` for D3D12, `set` for Vulkan
        SlangUInt registerIndex,   // `register` for D3D12, `binding` for Vulkan
        bool& outUsed);

    // Entry Point Reflection

    SLANG_API char const* spReflectionEntryPoint_getName(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API char const* spReflectionEntryPoint_getNameOverride(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API SlangReflectionFunction* spReflectionEntryPoint_getFunction(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API unsigned spReflectionEntryPoint_getParameterCount(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getParameterByIndex(
        SlangReflectionEntryPoint*  entryPoint,
        unsigned                    index);

    SLANG_API SlangStage spReflectionEntryPoint_getStage(SlangReflectionEntryPoint* entryPoint);

    SLANG_API void spReflectionEntryPoint_getComputeThreadGroupSize(
        SlangReflectionEntryPoint*  entryPoint,
        SlangUInt                   axisCount,
        SlangUInt*                  outSizeAlongAxis);

    SLANG_API void spReflectionEntryPoint_getComputeWaveSize(
        SlangReflectionEntryPoint* entryPoint,
        SlangUInt* outWaveSize);

    SLANG_API int spReflectionEntryPoint_usesAnySampleRateInput(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getVarLayout(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getResultVarLayout(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API int spReflectionEntryPoint_hasDefaultConstantBuffer(
        SlangReflectionEntryPoint* entryPoint);

    // SlangReflectionTypeParameter
    SLANG_API char const* spReflectionTypeParameter_GetName(SlangReflectionTypeParameter* typeParam);
    SLANG_API unsigned spReflectionTypeParameter_GetIndex(SlangReflectionTypeParameter* typeParam);
    SLANG_API unsigned spReflectionTypeParameter_GetConstraintCount(SlangReflectionTypeParameter* typeParam);
    SLANG_API SlangReflectionType* spReflectionTypeParameter_GetConstraintByIndex(SlangReflectionTypeParameter* typeParam, unsigned int index);

    // Shader Reflection

    SLANG_API unsigned spReflection_GetParameterCount(SlangReflection* reflection);
    SLANG_API SlangReflectionParameter* spReflection_GetParameterByIndex(SlangReflection* reflection, unsigned index);

    SLANG_API unsigned int spReflection_GetTypeParameterCount(SlangReflection* reflection);
    SLANG_API SlangReflectionTypeParameter* spReflection_GetTypeParameterByIndex(SlangReflection* reflection, unsigned int index);
    SLANG_API SlangReflectionTypeParameter* spReflection_FindTypeParameter(SlangReflection* reflection, char const* name);

    SLANG_API SlangReflectionType* spReflection_FindTypeByName(SlangReflection* reflection, char const* name);
    SLANG_API SlangReflectionTypeLayout* spReflection_GetTypeLayout(SlangReflection* reflection, SlangReflectionType* reflectionType, SlangLayoutRules rules);

    SLANG_API SlangReflectionFunction* spReflection_FindFunctionByName(SlangReflection* reflection, char const* name);

    SLANG_API SlangUInt spReflection_getEntryPointCount(SlangReflection* reflection);
    SLANG_API SlangReflectionEntryPoint* spReflection_getEntryPointByIndex(SlangReflection* reflection, SlangUInt index);
    SLANG_API SlangReflectionEntryPoint* spReflection_findEntryPointByName(SlangReflection* reflection, char const* name);

    SLANG_API SlangUInt spReflection_getGlobalConstantBufferBinding(SlangReflection* reflection);
    SLANG_API size_t spReflection_getGlobalConstantBufferSize(SlangReflection* reflection);

    SLANG_API  SlangReflectionType* spReflection_specializeType(
        SlangReflection*            reflection,
        SlangReflectionType*        type,
        SlangInt                    specializationArgCount,
        SlangReflectionType* const* specializationArgs,
        ISlangBlob**                outDiagnostics);

        /// Get the number of hashed strings
    SLANG_API SlangUInt spReflection_getHashedStringCount(
        SlangReflection*  reflection);

        /// Get a hashed string. The number of chars is written in outCount.
        /// The count does *NOT* including terminating 0. The returned string will be 0 terminated. 
    SLANG_API const char* spReflection_getHashedString(
        SlangReflection*  reflection,
        SlangUInt index,
        size_t* outCount);

        /// Compute a string hash.
        /// Count should *NOT* include terminating zero.
    SLANG_API SlangUInt32 spComputeStringHash(const char* chars, size_t count);

        /// Get a type layout representing reflection information for the global-scope prameters.
    SLANG_API SlangReflectionTypeLayout* spReflection_getGlobalParamsTypeLayout(
        SlangReflection* reflection);

        /// Get a variable layout representing reflection information for the global-scope prameters.
    SLANG_API SlangReflectionVariableLayout* spReflection_getGlobalParamsVarLayout(
        SlangReflection* reflection);

}
#ifdef __cplusplus

namespace slang
{
    struct ISession;
}

SLANG_API slang::ISession* spReflection_GetSession(SlangReflection* reflection);

/* Helper interfaces for C++ users */
namespace slang
{
    struct BufferReflection;
    struct TypeLayoutReflection;
    struct TypeReflection;
    struct VariableLayoutReflection;
    struct VariableReflection;
    
    struct UserAttribute
    {
        char const* getName()
        {
            return spReflectionUserAttribute_GetName((SlangReflectionUserAttribute*)this);
        }
        uint32_t getArgumentCount()
        {
            return (uint32_t)spReflectionUserAttribute_GetArgumentCount((SlangReflectionUserAttribute*)this);
        }
        TypeReflection* getArgumentType(uint32_t index)
        {
            return (TypeReflection*)spReflectionUserAttribute_GetArgumentType((SlangReflectionUserAttribute*)this, index);
        }
        SlangResult getArgumentValueInt(uint32_t index, int * value)
        {
            return spReflectionUserAttribute_GetArgumentValueInt((SlangReflectionUserAttribute*)this, index, value);
        }
        SlangResult getArgumentValueFloat(uint32_t index, float * value)
        {
            return spReflectionUserAttribute_GetArgumentValueFloat((SlangReflectionUserAttribute*)this, index, value);
        }
        const char* getArgumentValueString(uint32_t index, size_t * outSize)
        {
            return spReflectionUserAttribute_GetArgumentValueString((SlangReflectionUserAttribute*)this, index, outSize);
        }
    };

    struct TypeReflection
    {
        enum class Kind
        {
            None    = SLANG_TYPE_KIND_NONE,
            Struct  = SLANG_TYPE_KIND_STRUCT,
            Array   = SLANG_TYPE_KIND_ARRAY,
            Matrix  = SLANG_TYPE_KIND_MATRIX,
            Vector  = SLANG_TYPE_KIND_VECTOR,
            Scalar  = SLANG_TYPE_KIND_SCALAR,
            ConstantBuffer = SLANG_TYPE_KIND_CONSTANT_BUFFER,
            Resource = SLANG_TYPE_KIND_RESOURCE,
            SamplerState = SLANG_TYPE_KIND_SAMPLER_STATE,
            TextureBuffer = SLANG_TYPE_KIND_TEXTURE_BUFFER,
            ShaderStorageBuffer = SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER,
            ParameterBlock = SLANG_TYPE_KIND_PARAMETER_BLOCK,
            GenericTypeParameter = SLANG_TYPE_KIND_GENERIC_TYPE_PARAMETER,
            Interface = SLANG_TYPE_KIND_INTERFACE,
            OutputStream = SLANG_TYPE_KIND_OUTPUT_STREAM,
            Specialized = SLANG_TYPE_KIND_SPECIALIZED,
            Feedback = SLANG_TYPE_KIND_FEEDBACK,
            Pointer = SLANG_TYPE_KIND_POINTER,
        };

        enum ScalarType : SlangScalarTypeIntegral
        {
            None    = SLANG_SCALAR_TYPE_NONE,
            Void    = SLANG_SCALAR_TYPE_VOID,
            Bool    = SLANG_SCALAR_TYPE_BOOL,
            Int32   = SLANG_SCALAR_TYPE_INT32,
            UInt32  = SLANG_SCALAR_TYPE_UINT32,
            Int64   = SLANG_SCALAR_TYPE_INT64,
            UInt64  = SLANG_SCALAR_TYPE_UINT64,
            Float16 = SLANG_SCALAR_TYPE_FLOAT16,
            Float32 = SLANG_SCALAR_TYPE_FLOAT32,
            Float64 = SLANG_SCALAR_TYPE_FLOAT64,
            Int8    = SLANG_SCALAR_TYPE_INT8,
            UInt8   = SLANG_SCALAR_TYPE_UINT8,
            Int16   = SLANG_SCALAR_TYPE_INT16,
            UInt16  = SLANG_SCALAR_TYPE_UINT16,
        };

        Kind getKind()
        {
            return (Kind) spReflectionType_GetKind((SlangReflectionType*) this);
        }

        // only useful if `getKind() == Kind::Struct`
        unsigned int getFieldCount()
        {
            return spReflectionType_GetFieldCount((SlangReflectionType*) this);
        }

        VariableReflection* getFieldByIndex(unsigned int index)
        {
            return (VariableReflection*) spReflectionType_GetFieldByIndex((SlangReflectionType*) this, index);
        }

        bool isArray() { return getKind() == TypeReflection::Kind::Array; }

        TypeReflection* unwrapArray()
        {
            TypeReflection* type = this;
            while( type->isArray() )
            {
                type = type->getElementType();
            }
            return type;
        }

        // only useful if `getKind() == Kind::Array`
        size_t getElementCount()
        {
            return spReflectionType_GetElementCount((SlangReflectionType*) this);
        }

        size_t getTotalArrayElementCount()
        {
            if(!isArray()) return 0;
            size_t result = 1;
            TypeReflection* type = this;
            for(;;)
            {
                if(!type->isArray())
                    return result;

                result *= type->getElementCount();
                type = type->getElementType();
            }
        }

        TypeReflection* getElementType()
        {
            return (TypeReflection*) spReflectionType_GetElementType((SlangReflectionType*) this);
        }

        unsigned getRowCount()
        {
            return spReflectionType_GetRowCount((SlangReflectionType*) this);
        }

        unsigned getColumnCount()
        {
            return spReflectionType_GetColumnCount((SlangReflectionType*) this);
        }

        ScalarType getScalarType()
        {
            return (ScalarType) spReflectionType_GetScalarType((SlangReflectionType*) this);
        }

        TypeReflection* getResourceResultType()
        {
            return (TypeReflection*) spReflectionType_GetResourceResultType((SlangReflectionType*) this);
        }

        SlangResourceShape getResourceShape()
        {
            return spReflectionType_GetResourceShape((SlangReflectionType*) this);
        }

        SlangResourceAccess getResourceAccess()
        {
            return spReflectionType_GetResourceAccess((SlangReflectionType*) this);
        }

        char const* getName()
        {
            return spReflectionType_GetName((SlangReflectionType*) this);
        }

        SlangResult getFullName(ISlangBlob** outNameBlob)
        {
            return spReflectionType_GetFullName((SlangReflectionType*)this, outNameBlob);
        }

        unsigned int getUserAttributeCount()
        {
            return spReflectionType_GetUserAttributeCount((SlangReflectionType*)this);
        }
        UserAttribute* getUserAttributeByIndex(unsigned int index)
        {
            return (UserAttribute*)spReflectionType_GetUserAttribute((SlangReflectionType*)this, index);
        }
        UserAttribute* findUserAttributeByName(char const* name)
        {
            return (UserAttribute*)spReflectionType_FindUserAttributeByName((SlangReflectionType*)this, name);
        }
    };

    enum ParameterCategory : SlangParameterCategoryIntegral
    {
        // TODO: these aren't scoped...
        None = SLANG_PARAMETER_CATEGORY_NONE,
        Mixed = SLANG_PARAMETER_CATEGORY_MIXED,
        ConstantBuffer = SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
        ShaderResource = SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE,
        UnorderedAccess = SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS,
        VaryingInput = SLANG_PARAMETER_CATEGORY_VARYING_INPUT,
        VaryingOutput = SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT,
        SamplerState = SLANG_PARAMETER_CATEGORY_SAMPLER_STATE,
        Uniform = SLANG_PARAMETER_CATEGORY_UNIFORM,
        DescriptorTableSlot = SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT,
        SpecializationConstant = SLANG_PARAMETER_CATEGORY_SPECIALIZATION_CONSTANT,
        PushConstantBuffer = SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER,
        RegisterSpace = SLANG_PARAMETER_CATEGORY_REGISTER_SPACE,
        GenericResource = SLANG_PARAMETER_CATEGORY_GENERIC,

        RayPayload = SLANG_PARAMETER_CATEGORY_RAY_PAYLOAD,
        HitAttributes = SLANG_PARAMETER_CATEGORY_HIT_ATTRIBUTES,
        CallablePayload = SLANG_PARAMETER_CATEGORY_CALLABLE_PAYLOAD,

        ShaderRecord = SLANG_PARAMETER_CATEGORY_SHADER_RECORD,

        ExistentialTypeParam = SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM,
        ExistentialObjectParam = SLANG_PARAMETER_CATEGORY_EXISTENTIAL_OBJECT_PARAM,

        SubElementRegisterSpace = SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE,

        InputAttachmentIndex = SLANG_PARAMETER_CATEGORY_SUBPASS,

        MetalBuffer = SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
        MetalTexture = SLANG_PARAMETER_CATEGORY_METAL_TEXTURE,
        MetalArgumentBufferElement = SLANG_PARAMETER_CATEGORY_METAL_ARGUMENT_BUFFER_ELEMENT,
        MetalAttribute = SLANG_PARAMETER_CATEGORY_METAL_ATTRIBUTE,
        MetalPayload = SLANG_PARAMETER_CATEGORY_METAL_PAYLOAD,

        // DEPRECATED:
        VertexInput = SLANG_PARAMETER_CATEGORY_VERTEX_INPUT,
        FragmentOutput = SLANG_PARAMETER_CATEGORY_FRAGMENT_OUTPUT,
    };

    enum class BindingType : SlangBindingTypeIntegral
    {
        Unknown                             = SLANG_BINDING_TYPE_UNKNOWN,

        Sampler                             = SLANG_BINDING_TYPE_SAMPLER,
        Texture                             = SLANG_BINDING_TYPE_TEXTURE,
        ConstantBuffer                      = SLANG_BINDING_TYPE_CONSTANT_BUFFER,
        ParameterBlock                      = SLANG_BINDING_TYPE_PARAMETER_BLOCK,
        TypedBuffer                         = SLANG_BINDING_TYPE_TYPED_BUFFER,
        RawBuffer                           = SLANG_BINDING_TYPE_RAW_BUFFER,
        CombinedTextureSampler              = SLANG_BINDING_TYPE_COMBINED_TEXTURE_SAMPLER,
        InputRenderTarget                   = SLANG_BINDING_TYPE_INPUT_RENDER_TARGET,
        InlineUniformData                   = SLANG_BINDING_TYPE_INLINE_UNIFORM_DATA,
        RayTracingAccelerationStructure     = SLANG_BINDING_TYPE_RAY_TRACING_ACCELERATION_STRUCTURE,
        VaryingInput                        = SLANG_BINDING_TYPE_VARYING_INPUT,
        VaryingOutput                       = SLANG_BINDING_TYPE_VARYING_OUTPUT,
        ExistentialValue                    = SLANG_BINDING_TYPE_EXISTENTIAL_VALUE,
        PushConstant                        = SLANG_BINDING_TYPE_PUSH_CONSTANT,

        MutableFlag                         = SLANG_BINDING_TYPE_MUTABLE_FLAG,

        MutableTexture                      = SLANG_BINDING_TYPE_MUTABLE_TETURE,
        MutableTypedBuffer                  = SLANG_BINDING_TYPE_MUTABLE_TYPED_BUFFER,
        MutableRawBuffer                    = SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER,

        BaseMask                            = SLANG_BINDING_TYPE_BASE_MASK,
        ExtMask                             = SLANG_BINDING_TYPE_EXT_MASK,
    };

    struct TypeLayoutReflection
    {
        TypeReflection* getType()
        {
            return (TypeReflection*) spReflectionTypeLayout_GetType((SlangReflectionTypeLayout*) this);
        }

        TypeReflection::Kind getKind()
        {
            return (TypeReflection::Kind) spReflectionTypeLayout_getKind((SlangReflectionTypeLayout*) this);
        }

        size_t getSize(SlangParameterCategory category = SLANG_PARAMETER_CATEGORY_UNIFORM)
        {
            return spReflectionTypeLayout_GetSize((SlangReflectionTypeLayout*) this, category);
        }

        size_t getStride(SlangParameterCategory category = SLANG_PARAMETER_CATEGORY_UNIFORM)
        {
            return spReflectionTypeLayout_GetStride((SlangReflectionTypeLayout*) this, category);
        }

        int32_t getAlignment(SlangParameterCategory category = SLANG_PARAMETER_CATEGORY_UNIFORM)
        {
            return spReflectionTypeLayout_getAlignment((SlangReflectionTypeLayout*) this, category);
        }

        unsigned int getFieldCount()
        {
            return spReflectionTypeLayout_GetFieldCount((SlangReflectionTypeLayout*)this);
        }

        VariableLayoutReflection* getFieldByIndex(unsigned int index)
        {
            return (VariableLayoutReflection*) spReflectionTypeLayout_GetFieldByIndex((SlangReflectionTypeLayout*) this, index);
        }

        SlangInt findFieldIndexByName(char const* nameBegin, char const* nameEnd = nullptr)
        {
            return spReflectionTypeLayout_findFieldIndexByName((SlangReflectionTypeLayout*) this, nameBegin, nameEnd);
        }

        VariableLayoutReflection* getExplicitCounter()
        {
            return (VariableLayoutReflection*) spReflectionTypeLayout_GetExplicitCounter((SlangReflectionTypeLayout*) this);
        }

        bool isArray() { return getType()->isArray(); }

        TypeLayoutReflection* unwrapArray()
        {
            TypeLayoutReflection* typeLayout = this;
            while( typeLayout->isArray() )
            {
                typeLayout = typeLayout->getElementTypeLayout();
            }
            return typeLayout;
        }

        // only useful if `getKind() == Kind::Array`
        size_t getElementCount()
        {
            return getType()->getElementCount();
        }

        size_t getTotalArrayElementCount()
        {
            return getType()->getTotalArrayElementCount();
        }

        size_t getElementStride(SlangParameterCategory category)
        {
            return spReflectionTypeLayout_GetElementStride((SlangReflectionTypeLayout*) this, category);
        }

        TypeLayoutReflection* getElementTypeLayout()
        {
            return (TypeLayoutReflection*) spReflectionTypeLayout_GetElementTypeLayout((SlangReflectionTypeLayout*) this);
        }

        VariableLayoutReflection* getElementVarLayout()
        {
            return (VariableLayoutReflection*)spReflectionTypeLayout_GetElementVarLayout((SlangReflectionTypeLayout*) this);
        }

        VariableLayoutReflection* getContainerVarLayout()
        {
            return (VariableLayoutReflection*)spReflectionTypeLayout_getContainerVarLayout((SlangReflectionTypeLayout*) this);
        }

        // How is this type supposed to be bound?
        ParameterCategory getParameterCategory()
        {
            return (ParameterCategory) spReflectionTypeLayout_GetParameterCategory((SlangReflectionTypeLayout*) this);
        }

        unsigned int getCategoryCount()
        {
            return spReflectionTypeLayout_GetCategoryCount((SlangReflectionTypeLayout*) this);
        }

        ParameterCategory getCategoryByIndex(unsigned int index)
        {
            return (ParameterCategory) spReflectionTypeLayout_GetCategoryByIndex((SlangReflectionTypeLayout*) this, index);
        }

        unsigned getRowCount()
        {
            return getType()->getRowCount();
        }

        unsigned getColumnCount()
        {
            return getType()->getColumnCount();
        }

        TypeReflection::ScalarType getScalarType()
        {
            return getType()->getScalarType();
        }

        TypeReflection* getResourceResultType()
        {
            return getType()->getResourceResultType();
        }

        SlangResourceShape getResourceShape()
        {
            return getType()->getResourceShape();
        }

        SlangResourceAccess getResourceAccess()
        {
            return getType()->getResourceAccess();
        }

        char const* getName()
        {
            return getType()->getName();
        }

        SlangMatrixLayoutMode getMatrixLayoutMode()
        {
            return spReflectionTypeLayout_GetMatrixLayoutMode((SlangReflectionTypeLayout*) this);
        }

        int getGenericParamIndex()
        {
            return spReflectionTypeLayout_getGenericParamIndex(
                (SlangReflectionTypeLayout*) this);
        }

        TypeLayoutReflection* getPendingDataTypeLayout()
        {
            return (TypeLayoutReflection*) spReflectionTypeLayout_getPendingDataTypeLayout(
                (SlangReflectionTypeLayout*) this);
        }

        VariableLayoutReflection* getSpecializedTypePendingDataVarLayout()
        {
            return (VariableLayoutReflection*) spReflectionTypeLayout_getSpecializedTypePendingDataVarLayout(
                (SlangReflectionTypeLayout*) this);
        }

        SlangInt getBindingRangeCount()
        {
            return spReflectionTypeLayout_getBindingRangeCount(
                (SlangReflectionTypeLayout*) this);
        }

        BindingType getBindingRangeType(SlangInt index)
        {
            return (BindingType) spReflectionTypeLayout_getBindingRangeType(
                (SlangReflectionTypeLayout*) this,
                index);
        }

        bool isBindingRangeSpecializable(SlangInt index)
        {
            return (bool)spReflectionTypeLayout_isBindingRangeSpecializable(
                (SlangReflectionTypeLayout*)this,
                index);

        }

        SlangInt getBindingRangeBindingCount(SlangInt index)
        {
            return spReflectionTypeLayout_getBindingRangeBindingCount(
                (SlangReflectionTypeLayout*) this,
                index);
        }

        /*
        SlangInt getBindingRangeIndexOffset(SlangInt index)
        {
            return spReflectionTypeLayout_getBindingRangeIndexOffset(
                (SlangReflectionTypeLayout*) this,
                index);
        }

        SlangInt getBindingRangeSpaceOffset(SlangInt index)
        {
            return spReflectionTypeLayout_getBindingRangeSpaceOffset(
                (SlangReflectionTypeLayout*) this,
                index);
        }
        */

        SlangInt getFieldBindingRangeOffset(SlangInt fieldIndex)
        {
            return spReflectionTypeLayout_getFieldBindingRangeOffset(
                (SlangReflectionTypeLayout*) this,
                fieldIndex);
        }

        SlangInt getExplicitCounterBindingRangeOffset()
        {
            return spReflectionTypeLayout_getExplicitCounterBindingRangeOffset(
                (SlangReflectionTypeLayout*) this);
        }

        TypeLayoutReflection* getBindingRangeLeafTypeLayout(SlangInt index)
        {
            return (TypeLayoutReflection*) spReflectionTypeLayout_getBindingRangeLeafTypeLayout(
                (SlangReflectionTypeLayout*) this,
                index);
        }

        VariableReflection* getBindingRangeLeafVariable(SlangInt index)
        {
            return (VariableReflection*)spReflectionTypeLayout_getBindingRangeLeafVariable(
                (SlangReflectionTypeLayout*)this, index);
        }

        SlangInt getBindingRangeDescriptorSetIndex(SlangInt index)
        {
            return spReflectionTypeLayout_getBindingRangeDescriptorSetIndex(
                (SlangReflectionTypeLayout*) this,
                index);
        }

        SlangInt getBindingRangeFirstDescriptorRangeIndex(SlangInt index)
        {
            return spReflectionTypeLayout_getBindingRangeFirstDescriptorRangeIndex(
                (SlangReflectionTypeLayout*) this,
                index);
        }

        SlangInt getBindingRangeDescriptorRangeCount(SlangInt index)
        {
            return spReflectionTypeLayout_getBindingRangeDescriptorRangeCount(
                (SlangReflectionTypeLayout*) this,
                index);
        }

        SlangInt getDescriptorSetCount()
        {
            return spReflectionTypeLayout_getDescriptorSetCount(
                (SlangReflectionTypeLayout*) this);
        }

        SlangInt getDescriptorSetSpaceOffset(SlangInt setIndex)
        {
            return spReflectionTypeLayout_getDescriptorSetSpaceOffset(
                (SlangReflectionTypeLayout*) this,
                setIndex);
        }

        SlangInt getDescriptorSetDescriptorRangeCount(SlangInt setIndex)
        {
            return spReflectionTypeLayout_getDescriptorSetDescriptorRangeCount(
                (SlangReflectionTypeLayout*) this,
                setIndex);
        }

        SlangInt getDescriptorSetDescriptorRangeIndexOffset(SlangInt setIndex, SlangInt rangeIndex)
        {
            return spReflectionTypeLayout_getDescriptorSetDescriptorRangeIndexOffset(
                (SlangReflectionTypeLayout*) this,
                setIndex,
                rangeIndex);
        }

        SlangInt getDescriptorSetDescriptorRangeDescriptorCount(SlangInt setIndex, SlangInt rangeIndex)
        {
            return spReflectionTypeLayout_getDescriptorSetDescriptorRangeDescriptorCount(
                (SlangReflectionTypeLayout*) this,
                setIndex,
                rangeIndex);
        }

        BindingType getDescriptorSetDescriptorRangeType(SlangInt setIndex, SlangInt rangeIndex)
        {
            return (BindingType) spReflectionTypeLayout_getDescriptorSetDescriptorRangeType(
                (SlangReflectionTypeLayout*) this,
                setIndex,
                rangeIndex);
        }

        ParameterCategory getDescriptorSetDescriptorRangeCategory(SlangInt setIndex, SlangInt rangeIndex)
        {
            return (ParameterCategory) spReflectionTypeLayout_getDescriptorSetDescriptorRangeCategory(
                (SlangReflectionTypeLayout*) this,
                setIndex,
                rangeIndex);
        }

        SlangInt getSubObjectRangeCount()
        {
            return spReflectionTypeLayout_getSubObjectRangeCount(
                (SlangReflectionTypeLayout*) this);
        }

        SlangInt getSubObjectRangeBindingRangeIndex(SlangInt subObjectRangeIndex)
        {
            return spReflectionTypeLayout_getSubObjectRangeBindingRangeIndex(
                (SlangReflectionTypeLayout*) this,
                subObjectRangeIndex);
        }

        SlangInt getSubObjectRangeSpaceOffset(SlangInt subObjectRangeIndex)
        {
            return spReflectionTypeLayout_getSubObjectRangeSpaceOffset(
                (SlangReflectionTypeLayout*) this,
                subObjectRangeIndex);
        }

        VariableLayoutReflection* getSubObjectRangeOffset(SlangInt subObjectRangeIndex)
        {
            return (VariableLayoutReflection*) spReflectionTypeLayout_getSubObjectRangeOffset(
                (SlangReflectionTypeLayout*) this,
                subObjectRangeIndex);
        }
    };

    struct Modifier
    {
        enum ID : SlangModifierIDIntegral
        {
            Shared = SLANG_MODIFIER_SHARED,
            NoDiff = SLANG_MODIFIER_NO_DIFF,
            Static = SLANG_MODIFIER_STATIC,
            Const = SLANG_MODIFIER_CONST,
            Export = SLANG_MODIFIER_EXPORT,
            Extern = SLANG_MODIFIER_EXTERN,
            Differentiable = SLANG_MODIFIER_DIFFERENTIABLE,
        };
    };

    struct VariableReflection
    {
        char const* getName()
        {
            return spReflectionVariable_GetName((SlangReflectionVariable*) this);
        }

        TypeReflection* getType()
        {
            return (TypeReflection*) spReflectionVariable_GetType((SlangReflectionVariable*) this);
        }

        Modifier* findModifier(Modifier::ID id)
        {
            return (Modifier*) spReflectionVariable_FindModifier((SlangReflectionVariable*) this, (SlangModifierID) id);
        }

        unsigned int getUserAttributeCount()
        {
            return spReflectionVariable_GetUserAttributeCount((SlangReflectionVariable*)this);
        }
        UserAttribute* getUserAttributeByIndex(unsigned int index)
        {
            return (UserAttribute*)spReflectionVariable_GetUserAttribute((SlangReflectionVariable*)this, index);
        }
        UserAttribute* findUserAttributeByName(SlangSession* globalSession, char const* name)
        {
            return (UserAttribute*)spReflectionVariable_FindUserAttributeByName((SlangReflectionVariable*)this, globalSession, name);
        }
    };

    struct VariableLayoutReflection
    {
        VariableReflection* getVariable()
        {
            return (VariableReflection*) spReflectionVariableLayout_GetVariable((SlangReflectionVariableLayout*) this);
        }

        char const* getName()
        {
            return getVariable()->getName();
        }

        Modifier* findModifier(Modifier::ID id)
        {
            return getVariable()->findModifier(id);
        }

        TypeLayoutReflection* getTypeLayout()
        {
            return (TypeLayoutReflection*) spReflectionVariableLayout_GetTypeLayout((SlangReflectionVariableLayout*) this);
        }

        ParameterCategory getCategory()
        {
            return getTypeLayout()->getParameterCategory();
        }

        unsigned int getCategoryCount()
        {
            return getTypeLayout()->getCategoryCount();
        }

        ParameterCategory getCategoryByIndex(unsigned int index)
        {
            return getTypeLayout()->getCategoryByIndex(index);
        }


        size_t getOffset(SlangParameterCategory category = SLANG_PARAMETER_CATEGORY_UNIFORM)
        {
            return spReflectionVariableLayout_GetOffset((SlangReflectionVariableLayout*) this, category);
        }

        TypeReflection* getType()
        {
            return getVariable()->getType();
        }

        unsigned getBindingIndex()
        {
            return spReflectionParameter_GetBindingIndex((SlangReflectionVariableLayout*) this);
        }

        unsigned getBindingSpace()
        {
            return spReflectionParameter_GetBindingSpace((SlangReflectionVariableLayout*) this);
        }

        size_t getBindingSpace(SlangParameterCategory category)
        {
            return spReflectionVariableLayout_GetSpace((SlangReflectionVariableLayout*) this, category);
        }

        char const* getSemanticName()
        {
            return spReflectionVariableLayout_GetSemanticName((SlangReflectionVariableLayout*) this);
        }

        size_t getSemanticIndex()
        {
            return spReflectionVariableLayout_GetSemanticIndex((SlangReflectionVariableLayout*) this);
        }

        SlangStage getStage()
        {
            return spReflectionVariableLayout_getStage((SlangReflectionVariableLayout*) this);
        }

        VariableLayoutReflection* getPendingDataLayout()
        {
            return (VariableLayoutReflection*) spReflectionVariableLayout_getPendingDataLayout((SlangReflectionVariableLayout*) this);
        }
    };

    struct FunctionReflection
    {
        char const* getName()
        {
            return spReflectionFunction_GetName((SlangReflectionFunction*)this);
        }

        TypeReflection* getReturnType()
        {
            return (TypeReflection*)spReflectionFunction_GetResultType((SlangReflectionFunction*)this);
        }

        unsigned int getParameterCount()
        {
            return spReflectionFunction_GetParameterCount((SlangReflectionFunction*)this);
        }

        VariableReflection* getParameterByIndex(unsigned int index)
        {
            return (VariableReflection*)spReflectionFunction_GetParameter((SlangReflectionFunction*)this, index);
        }

        unsigned int getUserAttributeCount()
        {
            return spReflectionVariable_GetUserAttributeCount((SlangReflectionVariable*)this);
        }
        UserAttribute* getUserAttributeByIndex(unsigned int index)
        {
            return (UserAttribute*)spReflectionVariable_GetUserAttribute((SlangReflectionVariable*)this, index);
        }
        UserAttribute* findUserAttributeByName(SlangSession* globalSession, char const* name)
        {
            return (UserAttribute*)spReflectionVariable_FindUserAttributeByName((SlangReflectionVariable*)this, globalSession, name);
        }

        Modifier* findModifier(Modifier::ID id)
        {
            return (Modifier*)spReflectionVariable_FindModifier((SlangReflectionVariable*)this, (SlangModifierID)id);
        }
    };

    struct EntryPointReflection
    {
        char const* getName()
        {
            return spReflectionEntryPoint_getName((SlangReflectionEntryPoint*) this);
        }

        char const* getNameOverride()
        {
            return spReflectionEntryPoint_getNameOverride((SlangReflectionEntryPoint*)this);
        }

        unsigned getParameterCount()
        {
            return spReflectionEntryPoint_getParameterCount((SlangReflectionEntryPoint*) this);
        }

        FunctionReflection* getFunction()
        {
            return (FunctionReflection*)spReflectionEntryPoint_getFunction((SlangReflectionEntryPoint*) this);
        }

        VariableLayoutReflection* getParameterByIndex(unsigned index)
        {
            return (VariableLayoutReflection*) spReflectionEntryPoint_getParameterByIndex((SlangReflectionEntryPoint*) this, index);
        }

        SlangStage getStage()
        {
            return spReflectionEntryPoint_getStage((SlangReflectionEntryPoint*) this);
        }

        void getComputeThreadGroupSize(
            SlangUInt   axisCount,
            SlangUInt*  outSizeAlongAxis)
        {
            return spReflectionEntryPoint_getComputeThreadGroupSize((SlangReflectionEntryPoint*) this, axisCount, outSizeAlongAxis);
        }

        void getComputeWaveSize(
            SlangUInt* outWaveSize)
        {
            return spReflectionEntryPoint_getComputeWaveSize((SlangReflectionEntryPoint*)this, outWaveSize);
        }

        bool usesAnySampleRateInput()
        {
            return 0 != spReflectionEntryPoint_usesAnySampleRateInput((SlangReflectionEntryPoint*) this);
        }

        VariableLayoutReflection* getVarLayout()
        {
            return (VariableLayoutReflection*) spReflectionEntryPoint_getVarLayout((SlangReflectionEntryPoint*) this);
        }

        TypeLayoutReflection* getTypeLayout()
        {
            return getVarLayout()->getTypeLayout();
        }

        VariableLayoutReflection* getResultVarLayout()
        {
            return (VariableLayoutReflection*) spReflectionEntryPoint_getResultVarLayout((SlangReflectionEntryPoint*) this);
        }

        bool hasDefaultConstantBuffer()
        {
            return spReflectionEntryPoint_hasDefaultConstantBuffer((SlangReflectionEntryPoint*) this) != 0;
        }
    };

    typedef EntryPointReflection EntryPointLayout;

    struct TypeParameterReflection
    {
        char const* getName()
        {
            return spReflectionTypeParameter_GetName((SlangReflectionTypeParameter*) this);
        }
        unsigned getIndex()
        {
            return spReflectionTypeParameter_GetIndex((SlangReflectionTypeParameter*) this);
        }
        unsigned getConstraintCount()
        {
            return spReflectionTypeParameter_GetConstraintCount((SlangReflectionTypeParameter*) this);
        }
        TypeReflection* getConstraintByIndex(int index)
        {
            return (TypeReflection*)spReflectionTypeParameter_GetConstraintByIndex((SlangReflectionTypeParameter*) this, index);
        }
    };

    enum class LayoutRules : SlangLayoutRulesIntegral
    {
        Default = SLANG_LAYOUT_RULES_DEFAULT,
    };

    typedef struct ShaderReflection ProgramLayout;

    struct ShaderReflection
    {
        unsigned getParameterCount()
        {
            return spReflection_GetParameterCount((SlangReflection*) this);
        }

        unsigned getTypeParameterCount()
        {
            return spReflection_GetTypeParameterCount((SlangReflection*) this);
        }

        slang::ISession* getSession()
        {
            return spReflection_GetSession((SlangReflection*)this);
        }

        TypeParameterReflection* getTypeParameterByIndex(unsigned index)
        {
            return (TypeParameterReflection*)spReflection_GetTypeParameterByIndex((SlangReflection*) this, index);
        }

        TypeParameterReflection* findTypeParameter(char const* name)
        {
            return (TypeParameterReflection*)spReflection_FindTypeParameter((SlangReflection*)this, name);
        }

        VariableLayoutReflection* getParameterByIndex(unsigned index)
        {
            return (VariableLayoutReflection*) spReflection_GetParameterByIndex((SlangReflection*) this, index);
        }

        static ProgramLayout* get(SlangCompileRequest* request)
        {
            return (ProgramLayout*) spGetReflection(request);
        }

        SlangUInt getEntryPointCount()
        {
            return spReflection_getEntryPointCount((SlangReflection*) this);
        }

        EntryPointReflection* getEntryPointByIndex(SlangUInt index)
        {
            return (EntryPointReflection*) spReflection_getEntryPointByIndex((SlangReflection*) this, index);
        }

        SlangUInt getGlobalConstantBufferBinding()
        {
            return spReflection_getGlobalConstantBufferBinding((SlangReflection*)this);
        }

        size_t getGlobalConstantBufferSize()
        {
            return spReflection_getGlobalConstantBufferSize((SlangReflection*)this);
        }

        TypeReflection* findTypeByName(const char* name)
        {
            return (TypeReflection*)spReflection_FindTypeByName(
                (SlangReflection*) this,
                name);
        }

        FunctionReflection* findFunctionByName(const char* name)
        {
            return (FunctionReflection*)spReflection_FindFunctionByName(
                (SlangReflection*) this,
                name);
        }

        TypeLayoutReflection* getTypeLayout(
            TypeReflection* type,
            LayoutRules     rules = LayoutRules::Default)
        {
            return (TypeLayoutReflection*)spReflection_GetTypeLayout(
                (SlangReflection*) this,
                (SlangReflectionType*)type,
                SlangLayoutRules(rules));
        }

        EntryPointReflection* findEntryPointByName(const char* name)
        {
            return (EntryPointReflection*)spReflection_findEntryPointByName(
                (SlangReflection*) this,
                name);
        }

        TypeReflection* specializeType(
            TypeReflection*         type,
            SlangInt                specializationArgCount,
            TypeReflection* const*  specializationArgs,
            ISlangBlob**            outDiagnostics)
        {
            return (TypeReflection*) spReflection_specializeType(
                (SlangReflection*) this,
                (SlangReflectionType*) type,
                specializationArgCount,
                (SlangReflectionType* const*) specializationArgs,
                outDiagnostics);
        }

        SlangUInt getHashedStringCount() const { return spReflection_getHashedStringCount((SlangReflection*)this); }

        const char* getHashedString(SlangUInt index, size_t* outCount) const
        {
            return spReflection_getHashedString((SlangReflection*)this, index, outCount);
        }

        TypeLayoutReflection* getGlobalParamsTypeLayout()
        {
            return (TypeLayoutReflection*) spReflection_getGlobalParamsTypeLayout((SlangReflection*) this);
        }

        VariableLayoutReflection* getGlobalParamsVarLayout()
        {
            return (VariableLayoutReflection*) spReflection_getGlobalParamsVarLayout((SlangReflection*) this);
        }
    };

    typedef uint32_t CompileStdLibFlags;
    struct CompileStdLibFlag
    {
        enum Enum : CompileStdLibFlags
        {
            WriteDocumentation = 0x1,
        };
    };

    typedef ISlangBlob IBlob;

    struct IComponentType;
    struct ITypeConformance;
    struct IGlobalSession;
    struct IModule;

    struct SessionDesc;
    struct SpecializationArg;
    struct TargetDesc;

        /** A global session for interaction with the Slang library.

        An application may create and re-use a single global session across
        multiple sessions, in order to amortize startups costs (in current
        Slang this is mostly the cost of loading the Slang standard library).

        The global session is currently *not* thread-safe and objects created from
        a single global session should only be used from a single thread at
        a time.
        */
    struct IGlobalSession : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(0xc140b5fd, 0xc78, 0x452e, { 0xba, 0x7c, 0x1a, 0x1e, 0x70, 0xc7, 0xf7, 0x1c })

            /** Create a new session for loading and compiling code.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL createSession(
            SessionDesc const&  desc,
            ISession**          outSession) = 0;

            /** Look up the internal ID of a profile by its `name`.

            Profile IDs are *not* guaranteed to be stable across versions
            of the Slang library, so clients are expected to look up
            profiles by name at runtime.
            */
        virtual SLANG_NO_THROW SlangProfileID SLANG_MCALL findProfile(
            char const*     name) = 0;

            /** Set the path that downstream compilers (aka back end compilers) will
            be looked from.
            @param passThrough Identifies the downstream compiler
            @param path The path to find the downstream compiler (shared library/dll/executable)

            For back ends that are dlls/shared libraries, it will mean the path will
            be prefixed with the path when calls are made out to ISlangSharedLibraryLoader.
            For executables - it will look for executables along the path */
        virtual SLANG_NO_THROW void SLANG_MCALL setDownstreamCompilerPath(
            SlangPassThrough passThrough,
            char const* path) = 0;

            /** DEPRECATED: Use setLanguagePrelude

            Set the 'prelude' for generated code for a 'downstream compiler'.
            @param passThrough The downstream compiler for generated code that will have the prelude applied to it. 
            @param preludeText The text added pre-pended verbatim before the generated source

            That for pass-through usage, prelude is not pre-pended, preludes are for code generation only. 
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setDownstreamCompilerPrelude(
            SlangPassThrough passThrough,
            const char* preludeText) = 0;

            /** DEPRECATED: Use getLanguagePrelude

            Get the 'prelude' for generated code for a 'downstream compiler'.
            @param passThrough The downstream compiler for generated code that will have the prelude applied to it. 
            @param outPrelude  On exit holds a blob that holds the string of the prelude.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL getDownstreamCompilerPrelude(
            SlangPassThrough passThrough,
            ISlangBlob** outPrelude) = 0;

            /** Get the build version 'tag' string. The string is the same as produced via `git describe --tags`
            for the project. If Slang is built separately from the automated build scripts
            the contents will by default be 'unknown'. Any string can be set by changing the
            contents of 'slang-tag-version.h' file and recompiling the project.

            This method will return exactly the same result as the free function spGetBuildTagString.

            @return The build tag string
            */
        virtual SLANG_NO_THROW const char* SLANG_MCALL getBuildTagString() = 0;

            /* For a given source language set the default compiler.
            If a default cannot be chosen (for example the target cannot be achieved by the default),
            the default will not be used. 

            @param sourceLanguage the source language 
            @param defaultCompiler the default compiler for that language
            @return 
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL setDefaultDownstreamCompiler(
            SlangSourceLanguage sourceLanguage,
            SlangPassThrough defaultCompiler) = 0;

            /* For a source type get the default compiler 

            @param sourceLanguage the source language 
            @return The downstream compiler for that source language */
        virtual SlangPassThrough SLANG_MCALL getDefaultDownstreamCompiler(
            SlangSourceLanguage sourceLanguage) = 0;

            /* Set the 'prelude' placed before generated code for a specific language type.
            
            @param sourceLanguage The language the prelude should be inserted on.
            @param preludeText The text added pre-pended verbatim before the generated source

            Note! That for pass-through usage, prelude is not pre-pended, preludes are for code generation only. 
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setLanguagePrelude(
            SlangSourceLanguage sourceLanguage,
            const char* preludeText) = 0;

            /** Get the 'prelude' associated with a specific source language. 
            @param sourceLanguage The language the prelude should be inserted on.
            @param outPrelude  On exit holds a blob that holds the string of the prelude.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL getLanguagePrelude(
            SlangSourceLanguage sourceLanguage,
            ISlangBlob** outPrelude) = 0;

            /** Create a compile request.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL createCompileRequest(
            slang::ICompileRequest** outCompileRequest) = 0;

            /** Add new builtin declarations to be used in subsequent compiles.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL addBuiltins(
            char const*     sourcePath,
            char const*     sourceString) = 0;

            /** Set the session shared library loader. If this changes the loader, it may cause shared libraries to be unloaded
            @param loader The loader to set. Setting nullptr sets the default loader. 
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setSharedLibraryLoader(
            ISlangSharedLibraryLoader* loader) = 0;

            /** Gets the currently set shared library loader
            @return Gets the currently set loader. If returns nullptr, it's the default loader
            */
        virtual SLANG_NO_THROW ISlangSharedLibraryLoader* SLANG_MCALL getSharedLibraryLoader() = 0;

            /** Returns SLANG_OK if a the compilation target is supported for this session
            
            @param target The compilation target to test
            @return SLANG_OK if the target is available
            SLANG_E_NOT_IMPLEMENTED if not implemented in this build
            SLANG_E_NOT_FOUND if other resources (such as shared libraries) required to make target work could not be found
            SLANG_FAIL other kinds of failures */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL checkCompileTargetSupport(
            SlangCompileTarget  target) = 0;

            /** Returns SLANG_OK if a the pass through support is supported for this session
            @param session Session
            @param target The compilation target to test
            @return SLANG_OK if the target is available
            SLANG_E_NOT_IMPLEMENTED if not implemented in this build
            SLANG_E_NOT_FOUND if other resources (such as shared libraries) required to make target work could not be found
            SLANG_FAIL other kinds of failures */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL checkPassThroughSupport(
            SlangPassThrough    passThrough) = 0;

            /** Compile from (embedded source) the StdLib on the session.
            Will return a failure if there is already a StdLib available
            NOTE! API is experimental and not ready for production code
            @param flags to control compilation
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL compileStdLib(CompileStdLibFlags flags) = 0;

            /** Load the StdLib. Currently loads modules from the file system. 
            @param stdLib Start address of the serialized stdlib
            @param stdLibSizeInBytes The size in bytes of the serialized stdlib

            NOTE! API is experimental and not ready for production code
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadStdLib(const void* stdLib, size_t stdLibSizeInBytes) = 0;

            /** Save the StdLib modules to the file system
            @param archiveType The type of archive used to hold the stdlib
            @param outBlob The serialized blob containing the standard library

            NOTE! API is experimental and not ready for production code  */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL saveStdLib(SlangArchiveType archiveType, ISlangBlob** outBlob) = 0;

            /** Look up the internal ID of a capability by its `name`.

            Capability IDs are *not* guaranteed to be stable across versions
            of the Slang library, so clients are expected to look up
            capabilities by name at runtime.
            */
        virtual SLANG_NO_THROW SlangCapabilityID SLANG_MCALL findCapability(
            char const*     name) = 0;

            /** Set the downstream/pass through compiler to be used for a transition from the source type to the target type
            @param source The source 'code gen target'
            @param target The target 'code gen target'
            @param compiler The compiler/pass through to use for the transition from source to target
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setDownstreamCompilerForTransition(SlangCompileTarget source, SlangCompileTarget target, SlangPassThrough compiler) = 0;

            /** Get the downstream/pass through compiler for a transition specified by source and target
            @param source The source 'code gen target'
            @param target The target 'code gen target'
            @return The compiler that is used for the transition. Returns SLANG_PASS_THROUGH_NONE it is not defined
            */
        virtual SLANG_NO_THROW SlangPassThrough SLANG_MCALL getDownstreamCompilerForTransition(SlangCompileTarget source, SlangCompileTarget target) = 0;

            /** Get the time in seconds spent in the slang and downstream compiler.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL getCompilerElapsedTime(double* outTotalTime, double* outDownstreamTime) = 0;

            /** Specify a spirv.core.grammar.json file to load and use when
             * parsing and checking any SPIR-V code
             */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL setSPIRVCoreGrammar(
            char const* jsonPath) = 0;

            /** Parse slangc command line options into a SessionDesc that can be used to create a session
            *   with all the compiler options specified in the command line.
            *   @param argc The number of command line arguments.
            *   @param argv An input array of command line arguments to parse.
            *   @param outSessionDesc A pointer to a SessionDesc struct to receive parsed session desc.
            *   @param outAuxAllocation Auxillary memory allocated to hold data used in the sesion desc.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL parseCommandLineArguments(
            int argc, const char* const* argv, SessionDesc* outSessionDesc, ISlangUnknown** outAuxAllocation) = 0;

            /** Computes a digest that uniquely identifies the session description.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getSessionDescDigest(SessionDesc* sessionDesc, ISlangBlob** outBlob) = 0;
    };

    #define SLANG_UUID_IGlobalSession IGlobalSession::getTypeGuid()

    /*!
    @brief A request for one or more compilation actions to be performed.
    */
    struct ICompileRequest : public ISlangUnknown
    {
        SLANG_COM_INTERFACE( 0x96d33993, 0x317c, 0x4db5, { 0xaf, 0xd8, 0x66, 0x6e, 0xe7, 0x72, 0x48, 0xe2 } )
   
            /** Set the filesystem hook to use for a compile request

            The provided `fileSystem` will be used to load any files that
            need to be loaded during processing of the compile `request`.
            This includes:

              - Source files loaded via `spAddTranslationUnitSourceFile`
              - Files referenced via `#include`
              - Files loaded to resolve `#import` operations
                */
        virtual SLANG_NO_THROW void SLANG_MCALL setFileSystem(
            ISlangFileSystem*       fileSystem) = 0;

            /*!
            @brief Set flags to be used for compilation.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setCompileFlags(
            SlangCompileFlags       flags) = 0;

            /*!
            @brief Returns the compilation flags previously set with `setCompileFlags`
            */
        virtual SLANG_NO_THROW SlangCompileFlags SLANG_MCALL getCompileFlags() = 0;

            /*!
            @brief Set whether to dump intermediate results (for debugging) or not.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setDumpIntermediates(
            int                     enable) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setDumpIntermediatePrefix(
            const char* prefix) = 0;

            /*!
            @brief Set whether (and how) `#line` directives should be output.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setLineDirectiveMode(
            SlangLineDirectiveMode  mode) = 0;

            /*!
            @brief Sets the target for code generation.
            @param target The code generation target. Possible values are:
            - SLANG_GLSL. Generates GLSL code.
            - SLANG_HLSL. Generates HLSL code.
            - SLANG_SPIRV. Generates SPIR-V code.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setCodeGenTarget(
            SlangCompileTarget target) = 0;

            /*!
            @brief Add a code-generation target to be used.
            */
        virtual SLANG_NO_THROW int SLANG_MCALL addCodeGenTarget(
            SlangCompileTarget      target) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setTargetProfile(
            int                     targetIndex,
            SlangProfileID          profile) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setTargetFlags(
            int                     targetIndex,
            SlangTargetFlags        flags) = 0;


            /*!
            @brief Set the floating point mode (e.g., precise or fast) to use a target.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setTargetFloatingPointMode(
            int                     targetIndex,
            SlangFloatingPointMode  mode) = 0;

            /* DEPRECATED: use `spSetMatrixLayoutMode` instead. */
        virtual SLANG_NO_THROW void SLANG_MCALL setTargetMatrixLayoutMode(
            int                     targetIndex,
            SlangMatrixLayoutMode   mode) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setMatrixLayoutMode(
            SlangMatrixLayoutMode   mode) = 0;

            /*!
            @brief Set the level of debug information to produce.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setDebugInfoLevel(
            SlangDebugInfoLevel     level) = 0;

            /*!
            @brief Set the level of optimization to perform.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setOptimizationLevel(
            SlangOptimizationLevel  level) = 0;


    
            /*!
            @brief Set the container format to be used for binary output.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setOutputContainerFormat(
            SlangContainerFormat    format) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setPassThrough(
            SlangPassThrough        passThrough) = 0;

    
        virtual SLANG_NO_THROW void SLANG_MCALL setDiagnosticCallback(
            SlangDiagnosticCallback callback,
            void const*             userData) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setWriter(
            SlangWriterChannel      channel, 
            ISlangWriter*           writer) = 0;

        virtual SLANG_NO_THROW ISlangWriter* SLANG_MCALL getWriter(
            SlangWriterChannel      channel) = 0;

            /*!
            @brief Add a path to use when searching for referenced files.
            This will be used for both `#include` directives and also for explicit `__import` declarations.
            @param ctx The compilation context.
            @param searchDir The additional search directory.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL addSearchPath(
            const char*             searchDir) = 0;

            /*!
            @brief Add a macro definition to be used during preprocessing.
            @param key The name of the macro to define.
            @param value The value of the macro to define.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL addPreprocessorDefine(
            const char*             key,
            const char*             value) = 0;

            /*!
            @brief Set options using arguments as if specified via command line.
            @return Returns SlangResult. On success SLANG_SUCCEEDED(result) is true.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL processCommandLineArguments(
            char const* const*      args,
            int                     argCount) = 0;

            /** Add a distinct translation unit to the compilation request

            `name` is optional. 
            Returns the zero-based index of the translation unit created.
            */
        virtual SLANG_NO_THROW int SLANG_MCALL addTranslationUnit(
            SlangSourceLanguage     language,
            char const*             name) = 0;

    
            /** Set a default module name. Translation units will default to this module name if one is not
            passed. If not set each translation unit will get a unique name. 
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setDefaultModuleName(
            const char* defaultModuleName) = 0;

            /** Add a preprocessor definition that is scoped to a single translation unit.

            @param translationUnitIndex The index of the translation unit to get the definition.
            @param key The name of the macro to define.
            @param value The value of the macro to define.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitPreprocessorDefine(
            int                     translationUnitIndex,
            const char*             key,
            const char*             value) = 0;


            /** Add a source file to the given translation unit.

            If a user-defined file system has been specified via
            `spSetFileSystem`, then it will be used to load the
            file at `path`. Otherwise, Slang will use the OS
            file system.

            This function does *not* search for a file using
            the registered search paths (`spAddSearchPath`),
            and instead using the given `path` as-is.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceFile(
            int                     translationUnitIndex,
            char const*             path) = 0;

            /** Add a source string to the given translation unit.

            @param translationUnitIndex The index of the translation unit to add source to.
            @param path The file-system path that should be assumed for the source code.
            @param source A null-terminated UTF-8 encoded string of source code.

            The implementation will make a copy of the source code data.
            An application may free the buffer immediately after this call returns.

            The `path` will be used in any diagnostic output, as well
            as to determine the base path when resolving relative
            `#include`s.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceString(
            int                     translationUnitIndex,
            char const*             path,
            char const*             source) = 0;


            /** Add a slang library - such that its contents can be referenced during linking.
            This is equivalent to the -r command line option.

            @param basePath The base path used to lookup referenced modules.
            @param libData The library data
            @param libDataSize The size of the library data
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL addLibraryReference(
            const char* basePath,
            const void* libData,
            size_t libDataSize) = 0;

            /** Add a source string to the given translation unit.

            @param translationUnitIndex The index of the translation unit to add source to.
            @param path The file-system path that should be assumed for the source code.
            @param sourceBegin A pointer to a buffer of UTF-8 encoded source code.
            @param sourceEnd A pointer to to the end of the buffer specified in `sourceBegin`

            The implementation will make a copy of the source code data.
            An application may free the buffer immediately after this call returns.

            The `path` will be used in any diagnostic output, as well
            as to determine the base path when resolving relative
            `#include`s.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceStringSpan(
            int                     translationUnitIndex,
            char const*             path,
            char const*             sourceBegin,
            char const*             sourceEnd) = 0;

            /** Add a blob of source code to the given translation unit.

            @param translationUnitIndex The index of the translation unit to add source to.
            @param path The file-system path that should be assumed for the source code.
            @param sourceBlob A blob containing UTF-8 encoded source code.
            @param sourceEnd A pointer to to the end of the buffer specified in `sourceBegin`

            The compile request will retain a reference to the blob.

            The `path` will be used in any diagnostic output, as well
            as to determine the base path when resolving relative
            `#include`s.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceBlob(
            int                     translationUnitIndex,
            char const*             path,
            ISlangBlob*             sourceBlob) = 0;

            /** Add an entry point in a particular translation unit
            */
        virtual SLANG_NO_THROW int SLANG_MCALL addEntryPoint(
            int                     translationUnitIndex,
            char const*             name,
            SlangStage              stage) = 0;

            /** Add an entry point in a particular translation unit,
                with additional arguments that specify the concrete
                type names for entry-point generic type parameters.
            */
        virtual SLANG_NO_THROW int SLANG_MCALL addEntryPointEx(
            int                     translationUnitIndex,
            char const*             name,
            SlangStage              stage,
            int                     genericArgCount,
            char const**            genericArgs) = 0;

            /** Specify the arguments to use for global generic parameters.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL setGlobalGenericArgs(
            int                     genericArgCount,
            char const**            genericArgs) = 0;

            /** Specify the concrete type to be used for a global "existential slot."

            Every shader parameter (or leaf field of a `struct`-type shader parameter)
            that has an interface or array-of-interface type introduces an existential
            slot. The number of slots consumed by a shader parameter, and the starting
            slot of each parameter can be queried via the reflection API using
            `SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM`.

            In order to generate specialized code, a concrete type needs to be specified
            for each existential slot. This function specifies the name of the type
            (or in general a type *expression*) to use for a specific slot at the
            global scope.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL setTypeNameForGlobalExistentialTypeParam(
            int                     slotIndex,
            char const*             typeName) = 0;

            /** Specify the concrete type to be used for an entry-point "existential slot."

            Every shader parameter (or leaf field of a `struct`-type shader parameter)
            that has an interface or array-of-interface type introduces an existential
            slot. The number of slots consumed by a shader parameter, and the starting
            slot of each parameter can be queried via the reflection API using
            `SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM`.

            In order to generate specialized code, a concrete type needs to be specified
            for each existential slot. This function specifies the name of the type
            (or in general a type *expression*) to use for a specific slot at the
            entry-point scope.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL setTypeNameForEntryPointExistentialTypeParam(
            int                     entryPointIndex,
            int                     slotIndex,
            char const*             typeName) = 0;

            /** Enable or disable an experimental, best-effort GLSL frontend
             */
        virtual SLANG_NO_THROW void SLANG_MCALL setAllowGLSLInput(
            bool                    value) = 0;

            /** Execute the compilation request.

            @returns  SlangResult, SLANG_OK on success. Use SLANG_SUCCEEDED() and SLANG_FAILED() to test SlangResult.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL compile() = 0;


            /** Get any diagnostic messages reported by the compiler.

            @returns A null-terminated UTF-8 encoded string of diagnostic messages.

            The returned pointer is only guaranteed to be valid
            until `request` is destroyed. Applications that wish to
            hold on to the diagnostic output for longer should use
            `getDiagnosticOutputBlob`.
            */
        virtual SLANG_NO_THROW char const* SLANG_MCALL getDiagnosticOutput() = 0;

            /** Get diagnostic messages reported by the compiler.

            @param outBlob A pointer to receive a blob holding a nul-terminated UTF-8 encoded string of diagnostic messages.
            @returns A `SlangResult` indicating success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getDiagnosticOutputBlob(
            ISlangBlob**            outBlob) = 0;


            /** Get the number of files that this compilation depended on.

            This includes both the explicit source files, as well as any
            additional files that were transitively referenced (e.g., via
            a `#include` directive).
            */
        virtual SLANG_NO_THROW int SLANG_MCALL getDependencyFileCount() = 0;

            /** Get the path to a file this compilation depended on.
            */
        virtual SLANG_NO_THROW char const* SLANG_MCALL getDependencyFilePath(
            int                     index) = 0;

            /** Get the number of translation units associated with the compilation request
            */
        virtual SLANG_NO_THROW int SLANG_MCALL getTranslationUnitCount() = 0;

            /** Get the output source code associated with a specific entry point.

            The lifetime of the output pointer is the same as `request`.
            */
        virtual SLANG_NO_THROW char const* SLANG_MCALL getEntryPointSource(
            int                     entryPointIndex) = 0;

            /** Get the output bytecode associated with a specific entry point.

            The lifetime of the output pointer is the same as `request`.
            */
        virtual SLANG_NO_THROW void const* SLANG_MCALL getEntryPointCode(
            int                     entryPointIndex,
            size_t*                 outSize) = 0;

            /** Get the output code associated with a specific entry point.

            @param entryPointIndex The index of the entry point to get code for.
            @param targetIndex The index of the target to get code for (default: zero).
            @param outBlob A pointer that will receive the blob of code
            @returns A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCodeBlob(
            int                     entryPointIndex,
            int                     targetIndex,
            ISlangBlob**            outBlob) = 0;

            /** Get entry point 'callable' functions accessible through the ISlangSharedLibrary interface.

            That the functions remain in scope as long as the ISlangSharedLibrary interface is in scope.

            NOTE! Requires a compilation target of SLANG_HOST_CALLABLE.
    
            @param entryPointIndex  The index of the entry point to get code for.
            @param targetIndex      The index of the target to get code for (default: zero).
            @param outSharedLibrary A pointer to a ISharedLibrary interface which functions can be queried on.
            @returns                A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
            int                     entryPointIndex,
            int                     targetIndex,
            ISlangSharedLibrary**   outSharedLibrary) = 0;

            /** Get the output code associated with a specific target.

            @param targetIndex The index of the target to get code for (default: zero).
            @param outBlob A pointer that will receive the blob of code
            @returns A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTargetCodeBlob(
            int                     targetIndex,
            ISlangBlob**            outBlob) = 0;

            /** Get 'callable' functions for a target accessible through the ISlangSharedLibrary interface.

            That the functions remain in scope as long as the ISlangSharedLibrary interface is in scope.

            NOTE! Requires a compilation target of SLANG_HOST_CALLABLE.
    
            @param targetIndex      The index of the target to get code for (default: zero).
            @param outSharedLibrary A pointer to a ISharedLibrary interface which functions can be queried on.
            @returns                A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTargetHostCallable(
            int                     targetIndex,
            ISlangSharedLibrary**   outSharedLibrary) = 0;

            /** Get the output bytecode associated with an entire compile request.

            The lifetime of the output pointer is the same as `request` and the last spCompile.

            @param outSize          The size of the containers contents in bytes. Will be zero if there is no code available.
            @returns                Pointer to start of the contained data, or nullptr if there is no code available.
            */
        virtual SLANG_NO_THROW void const* SLANG_MCALL getCompileRequestCode(
            size_t*                 outSize) = 0;

            /** Get the compilation result as a file system.
            The result is not written to the actual OS file system, but is made avaiable as an 
            in memory representation.
            */
        virtual SLANG_NO_THROW ISlangMutableFileSystem* SLANG_MCALL getCompileRequestResultAsFileSystem() = 0;

            /** Return the container code as a blob. The container blob is created as part of a compilation (with spCompile),
            and a container is produced with a suitable ContainerFormat. 

            @param outSize          The blob containing the container data. 
            @returns                A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getContainerCode(
            ISlangBlob**            outBlob) = 0;

            /** Load repro from memory specified.

            Should only be performed on a newly created request.

            NOTE! When using the fileSystem, files will be loaded via their `unique names` as if they are part of the flat file system. This
            mechanism is described more fully in docs/repro.md.

            @param fileSystem       An (optional) filesystem. Pass nullptr to just use contents of repro held in data.
            @param data             The data to load from.
            @param size             The size of the data to load from. 
            @returns                A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadRepro(
            ISlangFileSystem* fileSystem,
            const void* data,
            size_t size) = 0;

            /** Save repro state. Should *typically* be performed after spCompile, so that everything
            that is needed for a compilation is available. 

            @param outBlob          Blob that will hold the serialized state
            @returns                A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL saveRepro(
            ISlangBlob** outBlob) = 0;

            /** Enable repro capture.

            Should be set after any ISlangFileSystem has been set, but before any compilation. It ensures that everything
            that the ISlangFileSystem accesses will be correctly recorded.
            Note that if a ISlangFileSystem/ISlangFileSystemExt isn't explicitly set (ie the default is used), then the
            request will automatically be set up to record everything appropriate. 

            @returns                A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL enableReproCapture() = 0;

            /** Get the (linked) program for a compile request.

            The linked program will include all of the global-scope modules for the
            translation units in the program, plus any modules that they `import`
            (transitively), specialized to any global specialization arguments that
            were provided via the API.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getProgram(
            slang::IComponentType** outProgram) = 0;

            /** Get the (partially linked) component type for an entry point.

            The returned component type will include the entry point at the
            given index, and will be specialized using any specialization arguments
            that were provided for it via the API.

            The returned component will *not* include the modules representing
            the global scope and its dependencies/specialization, so a client
            program will typically want to compose this component type with
            the one returned by `spCompileRequest_getProgram` to get a complete
            and usable component type from which kernel code can be requested.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPoint(
            SlangInt                entryPointIndex,
            slang::IComponentType** outEntryPoint) = 0;

            /** Get the (un-linked) module for a translation unit.

            The returned module will not be linked against any dependencies,
            nor against any entry points (even entry points declared inside
            the module). Similarly, the module will not be specialized
            to the arguments that might have been provided via the API.

            This function provides an atomic unit of loaded code that
            is suitable for looking up types and entry points in the
            given module, and for linking together to produce a composite
            program that matches the needs of an application.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getModule(
            SlangInt                translationUnitIndex,
            slang::IModule**        outModule) = 0;

            /** Get the `ISession` handle behind the `SlangCompileRequest`.
            TODO(JS): Arguably this should just return the session pointer.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getSession(
            slang::ISession** outSession) = 0;

            /** get reflection data from a compilation request */
        virtual SLANG_NO_THROW SlangReflection* SLANG_MCALL getReflection() = 0;

            /** Make output specially handled for command line output */
        virtual SLANG_NO_THROW void SLANG_MCALL setCommandLineCompilerMode() = 0;

            /** Add a defined capability that should be assumed available on the target */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL addTargetCapability(
            SlangInt            targetIndex,
            SlangCapabilityID   capability) = 0;

            /** Get the (linked) program for a compile request, including all entry points.

            The resulting program will include all of the global-scope modules for the
            translation units in the program, plus any modules that they `import`
            (transitively), specialized to any global specialization arguments that
            were provided via the API, as well as all entry points specified for compilation,
            specialized to their entry-point specialization arguments.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getProgramWithEntryPoints(
            slang::IComponentType** outProgram) = 0;

        virtual SLANG_NO_THROW SlangResult SLANG_MCALL isParameterLocationUsed(
            SlangInt entryPointIndex,
            SlangInt targetIndex,
            SlangParameterCategory category,
            SlangUInt spaceIndex,
            SlangUInt registerIndex,
            bool& outUsed) = 0;

            /** Set the line directive mode for a target.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setTargetLineDirectiveMode(
            SlangInt targetIndex,
            SlangLineDirectiveMode mode) = 0;

            /** Set whether to use scalar buffer layouts for GLSL/Vulkan targets.
                If true, the generated GLSL/Vulkan code will use `scalar` layout for storage buffers.
                If false, the resulting code will std430 for storage buffers.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL setTargetForceGLSLScalarBufferLayout(int targetIndex, bool forceScalarLayout) = 0;

            /** Overrides the severity of a specific diagnostic message.

            @param messageID            Numeric identifier of the message to override,
                                        as defined in the 1st parameter of the DIAGNOSTIC macro.
            @param overrideSeverity     New severity of the message. If the message is originally Error or Fatal,
                                        the new severity cannot be lower than that.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL overrideDiagnosticSeverity(
            SlangInt messageID,
            SlangSeverity overrideSeverity) = 0;

            /** Returns the currently active flags of the request's diagnostic sink. */
        virtual SLANG_NO_THROW SlangDiagnosticFlags SLANG_MCALL getDiagnosticFlags() = 0;

            /** Sets the flags of the request's diagnostic sink.
                The previously specified flags are discarded. */
        virtual SLANG_NO_THROW void SLANG_MCALL setDiagnosticFlags(SlangDiagnosticFlags flags) = 0;

            /** Set the debug format to be used for debugging information */
        virtual SLANG_NO_THROW void SLANG_MCALL setDebugInfoFormat(SlangDebugInfoFormat debugFormat) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setEnableEffectAnnotations(bool value) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setReportDownstreamTime(bool value) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setReportPerfBenchmark(bool value) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setSkipSPIRVValidation(bool value) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setTargetUseMinimumSlangOptimization(int targetIndex, bool value) = 0;

        virtual SLANG_NO_THROW void SLANG_MCALL setIgnoreCapabilityCheck(bool value) = 0;

        // return a copy of internal profiling results, and if `shouldClear` is true, clear the internal profiling results before returning.
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getCompileTimeProfile(ISlangProfiler** compileTimeProfile, bool shouldClear) = 0;

    };

    #define SLANG_UUID_ICompileRequest ICompileRequest::getTypeGuid()

        /** Description of a code generation target.
        */
    struct TargetDesc
    {
            /** The size of this structure, in bytes.
            */
        size_t structureSize = sizeof(TargetDesc);

            /** The target format to generate code for (e.g., SPIR-V, DXIL, etc.)
            */
        SlangCompileTarget      format = SLANG_TARGET_UNKNOWN;

            /** The compilation profile supported by the target (e.g., "Shader Model 5.1")
            */
        SlangProfileID          profile = SLANG_PROFILE_UNKNOWN;

            /** Flags for the code generation target. Currently unused. */
        SlangTargetFlags        flags = kDefaultTargetFlags;

            /** Default mode to use for floating-point operations on the target.
            */
        SlangFloatingPointMode  floatingPointMode = SLANG_FLOATING_POINT_MODE_DEFAULT;

            /** The line directive mode for output source code.
            */
        SlangLineDirectiveMode lineDirectiveMode = SLANG_LINE_DIRECTIVE_MODE_DEFAULT;

            /** Whether to force `scalar` layout for glsl shader storage buffers.
            */
        bool forceGLSLScalarBufferLayout = false;

            /** Pointer to an array of compiler option entries, whose size is compilerOptionEntryCount.
            */
        CompilerOptionEntry* compilerOptionEntries = nullptr;

            /** Number of additional compiler option entries.
            */
        uint32_t compilerOptionEntryCount = 0;

    };

    typedef uint32_t SessionFlags;
    enum
    {
        kSessionFlags_None = 0
    };

    struct PreprocessorMacroDesc
    {
        const char* name;
        const char* value;
    };

    struct SessionDesc
    {
            /** The size of this structure, in bytes.
             */
        size_t structureSize = sizeof(SessionDesc);

            /** Code generation targets to include in the session.
            */
        TargetDesc const*   targets = nullptr;
        SlangInt            targetCount = 0;

            /** Flags to configure the session.
            */
        SessionFlags flags = kSessionFlags_None;

            /** Default layout to assume for variables with matrix types.
            */
        SlangMatrixLayoutMode defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

            /** Paths to use when searching for `#include`d or `import`ed files.
            */
        char const* const*  searchPaths = nullptr;
        SlangInt            searchPathCount = 0;

        PreprocessorMacroDesc const*    preprocessorMacros = nullptr;
        SlangInt                        preprocessorMacroCount = 0;

        ISlangFileSystem* fileSystem = nullptr;

        bool enableEffectAnnotations = false;
        bool allowGLSLSyntax = false;

        /** Pointer to an array of compiler option entries, whose size is compilerOptionEntryCount.
        */
        CompilerOptionEntry* compilerOptionEntries = nullptr;

        /** Number of additional compiler option entries.
        */
        uint32_t compilerOptionEntryCount = 0;

    };

    enum class ContainerType
    {
        None, UnsizedArray, StructuredBuffer, ConstantBuffer, ParameterBlock
    };

        /** A session provides a scope for code that is loaded.

        A session can be used to load modules of Slang source code,
        and to request target-specific compiled binaries and layout
        information.

        In order to be able to load code, the session owns a set
        of active "search paths" for resolving `#include` directives
        and `import` declrations, as well as a set of global
        preprocessor definitions that will be used for all code
        that gets `import`ed in the session.

        If multiple user shaders are loaded in the same session,
        and import the same module (e.g., two source files do `import X`)
        then there will only be one copy of `X` loaded within the session.

        In order to be able to generate target code, the session
        owns a list of available compilation targets, which specify
        code generation options.

        Code loaded and compiled within a session is owned by the session
        and will remain resident in memory until the session is released.
        Applications wishing to control the memory usage for compiled
        and loaded code should use multiple sessions.
        */
    struct ISession : public ISlangUnknown
    {
        SLANG_COM_INTERFACE( 0x67618701, 0xd116, 0x468f, { 0xab, 0x3b, 0x47, 0x4b, 0xed, 0xce, 0xe, 0x3d } )

            /** Get the global session thas was used to create this session.
            */
        virtual SLANG_NO_THROW IGlobalSession* SLANG_MCALL getGlobalSession() = 0;

            /** Load a module as it would be by code using `import`.
            */
        virtual SLANG_NO_THROW IModule* SLANG_MCALL loadModule(
            const char* moduleName,
            IBlob**     outDiagnostics = nullptr) = 0;

            /** Load a module from Slang source code.
            */
        virtual SLANG_NO_THROW IModule* SLANG_MCALL loadModuleFromSource(
            const char* moduleName,
            const char* path,
            slang::IBlob* source,
            slang::IBlob** outDiagnostics = nullptr) = 0;

            /** Combine multiple component types to create a composite component type.

            The `componentTypes` array must contain `componentTypeCount` pointers
            to component types that were loaded or created using the same session.

            The shader parameters and specialization parameters of the composite will
            be the union of those in `componentTypes`. The relative order of child
            component types is significant, and will affect the order in which
            parameters are reflected and laid out.

            The entry-point functions of the composite will be the union of those in
            `componentTypes`, and will follow the ordering of `componentTypes`.

            The requirements of the composite component type will be a subset of
            those in `componentTypes`. If an entry in `componentTypes` has a requirement
            that can be satisfied by another entry, then the composition will
            satisfy the requirement and it will not appear as a requirement of
            the composite. If multiple entries in `componentTypes` have a requirement
            for the same type, then only the first such requirement will be retained
            on the composite. The relative ordering of requirements on the composite
            will otherwise match that of `componentTypes`.

            If any diagnostics are generated during creation of the composite, they
            will be written to `outDiagnostics`. If an error is encountered, the
            function will return null.

            It is an error to create a composite component type that recursively
            aggregates the a single module more than once.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL createCompositeComponentType(
            IComponentType* const*  componentTypes,
            SlangInt                componentTypeCount,
            IComponentType**        outCompositeComponentType,
            ISlangBlob**            outDiagnostics = nullptr) = 0;

            /** Specialize a type based on type arguments.
            */
        virtual SLANG_NO_THROW TypeReflection* SLANG_MCALL specializeType(
            TypeReflection*             type,
            SpecializationArg const*    specializationArgs,
            SlangInt                    specializationArgCount,
            ISlangBlob**                outDiagnostics = nullptr) = 0;


            /** Get the layout `type` on the chosen `target`.
            */
        virtual SLANG_NO_THROW TypeLayoutReflection* SLANG_MCALL getTypeLayout(
            TypeReflection* type,
            SlangInt        targetIndex = 0,
            LayoutRules     rules = LayoutRules::Default,
            ISlangBlob**    outDiagnostics = nullptr) = 0;

            /** Get a container type from `elementType`. For example, given type `T`, returns
                a type that represents `StructuredBuffer<T>`.

                @param `elementType`: the element type to wrap around.
                @param `containerType`: the type of the container to wrap `elementType` in.
                @param `outDiagnostics`: a blob to receive diagnostic messages.
            */
        virtual SLANG_NO_THROW TypeReflection* SLANG_MCALL getContainerType(
            TypeReflection* elementType,
            ContainerType containerType,
            ISlangBlob** outDiagnostics = nullptr) = 0;

            /** Return a `TypeReflection` that represents the `__Dynamic` type.
                This type can be used as a specialization argument to indicate using
                dynamic dispatch.
            */
        virtual SLANG_NO_THROW TypeReflection* SLANG_MCALL getDynamicType() = 0;

            /** Get the mangled name for a type RTTI object.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTypeRTTIMangledName(
            TypeReflection* type,
            ISlangBlob** outNameBlob) = 0;

            /** Get the mangled name for a type witness.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTypeConformanceWitnessMangledName(
            TypeReflection* type,
            TypeReflection* interfaceType,
            ISlangBlob** outNameBlob) = 0;

            /** Get the sequential ID used to identify a type witness in a dynamic object.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTypeConformanceWitnessSequentialID(
            slang::TypeReflection* type,
            slang::TypeReflection* interfaceType,
            uint32_t*              outId) = 0;

            /** Create a request to load/compile front-end code.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL createCompileRequest(
            SlangCompileRequest**   outCompileRequest) = 0;

        
            /** Creates a `IComponentType` that represents a type's conformance to an interface.
                The retrieved `ITypeConformance` objects can be included in a composite `IComponentType`
                to explicitly specify which implementation types should be included in the final compiled
                code. For example, if an module defines `IMaterial` interface and `AMaterial`,
                `BMaterial`, `CMaterial` types that implements the interface, the user can exclude
                `CMaterial` implementation from the resulting shader code by explcitly adding
                `AMaterial:IMaterial` and `BMaterial:IMaterial` conformances to a composite
                `IComponentType` and get entry point code from it. The resulting code will not have
                anything related to `CMaterial` in the dynamic dispatch logic. If the user does not
                explicitly include any `TypeConformances` to an interface type, all implementations to
                that interface will be included by default. By linking a `ITypeConformance`, the user is
                also given the opportunity to specify the dispatch ID of the implementation type. If
                `conformanceIdOverride` is -1, there will be no override behavior and Slang will
                automatically assign IDs to implementation types. The automatically assigned IDs can be
                queried via `ISession::getTypeConformanceWitnessSequentialID`.

                Returns SLANG_OK if succeeds, or SLANG_FAIL if `type` does not conform to `interfaceType`.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL createTypeConformanceComponentType(
            slang::TypeReflection* type,
            slang::TypeReflection* interfaceType,
            ITypeConformance** outConformance,
            SlangInt conformanceIdOverride,
            ISlangBlob** outDiagnostics) = 0;

            /** Load a module from a Slang module blob.
            */
        virtual SLANG_NO_THROW IModule* SLANG_MCALL loadModuleFromIRBlob(
            const char* moduleName,
            const char* path,
            slang::IBlob* source,
            slang::IBlob** outDiagnostics = nullptr) = 0;

        virtual SLANG_NO_THROW SlangInt SLANG_MCALL getLoadedModuleCount() = 0;
        virtual SLANG_NO_THROW IModule* SLANG_MCALL getLoadedModule(SlangInt index) = 0;

            /** Checks if a precompiled binary module is up-to-date with the current compiler
            *   option settings and the source file contents.
            */
        virtual SLANG_NO_THROW bool SLANG_MCALL isBinaryModuleUpToDate(
            const char* modulePath, slang::IBlob* binaryModuleBlob) = 0;

            /** Load a module from a string.
            */
        virtual SLANG_NO_THROW IModule* SLANG_MCALL loadModuleFromSourceString(
            const char* moduleName,
            const char* path,
            const char* string,
            slang::IBlob** outDiagnostics = nullptr) = 0;
    };

    #define SLANG_UUID_ISession ISession::getTypeGuid()

        /** A component type is a unit of shader code layout, reflection, and linking.

        A component type is a unit of shader code that can be included into
        a linked and compiled shader program. Each component type may have:

        * Zero or more uniform shader parameters, representing textures,
          buffers, etc. that the code in the component depends on.

        * Zero or more *specialization* parameters, which are type or
          value parameters that can be used to synthesize specialized
          versions of the component type.

        * Zero or more entry points, which are the individually invocable
          kernels that can have final code generated.

        * Zero or more *requirements*, which are other component
          types on which the component type depends.

        One example of a component type is a module of Slang code:

        * The global-scope shader parameters declared in the module are
          the parameters when considered as a component type.

        * Any global-scope generic or interface type parameters introduce
          specialization parameters for the module.

        * A module does not by default include any entry points when
          considered as a component type (although the code of the
          module might *declare* some entry points).

        * Any other modules that are `import`ed in the source code
          become requirements of the module, when considered as a
          component type.

        An entry point is another example of a component type:

        * The `uniform` parameters of the entry point function are
          its shader parameters when considered as a component type.

        * Any generic or interface-type parameters of the entry point
          introduce specialization parameters.

        * An entry point component type exposes a single entry point (itself).

        * An entry point has one requirement for the module in which
          it was defined.

        Component types can be manipulated in a few ways:

        * Multiple component types can be combined into a composite, which
          combines all of their code, parameters, etc.

        * A component type can be specialized, by "plugging in" types and
          values for its specialization parameters.

        * A component type can be laid out for a particular target, giving
          offsets/bindings to the shader parameters it contains.

        * Generated kernel code can be requested for entry points.

        */
    struct IComponentType : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(0x5bc42be8, 0x5c50, 0x4929, { 0x9e, 0x5e, 0xd1, 0x5e, 0x7c, 0x24, 0x1, 0x5f })

            /** Get the runtime session that this component type belongs to.
            */
        virtual SLANG_NO_THROW ISession* SLANG_MCALL getSession() = 0;

            /** Get the layout for this program for the chosen `targetIndex`.

            The resulting layout will establish offsets/bindings for all
            of the global and entry-point shader parameters in the
            component type.

            If this component type has specialization parameters (that is,
            it is not fully specialized), then the resulting layout may
            be incomplete, and plugging in arguments for generic specialization
            parameters may result in a component type that doesn't have
            a compatible layout. If the component type only uses
            interface-type specialization parameters, then the layout
            for a specialization should be compatible with an unspecialized
            layout (all parameters in the unspecialized layout will have
            the same offset/binding in the specialized layout).

            If this component type is combined into a composite, then
            the absolute offsets/bindings of parameters may not stay the same.
            If the shader parameters in a component type don't make
            use of explicit binding annotations (e.g., `register(...)`),
            then the *relative* offset of shader parameters will stay
            the same when it is used in a composition.
            */
        virtual SLANG_NO_THROW ProgramLayout* SLANG_MCALL getLayout(
            SlangInt    targetIndex = 0,
            IBlob**     outDiagnostics = nullptr) = 0;

            /** Get the number of (unspecialized) specialization parameters for the component type.
            */
        virtual SLANG_NO_THROW SlangInt SLANG_MCALL getSpecializationParamCount() = 0;

            /** Get the compiled code for the entry point at `entryPointIndex` for the chosen `targetIndex`

            Entry point code can only be computed for a component type that
            has no specialization parameters (it must be fully specialized)
            and that has no requirements (it must be fully linked).

            If code has not already been generated for the given entry point and target,
            then a compilation error may be detected, in which case `outDiagnostics`
            (if non-null) will be filled in with a blob of messages diagnosing the error.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
            SlangInt    entryPointIndex,
            SlangInt    targetIndex,
            IBlob**     outCode,
            IBlob**     outDiagnostics = nullptr) = 0;

            /** Get the compilation result as a file system.

            Has the same requirements as getEntryPointCode.

            The result is not written to the actual OS file system, but is made avaiable as an
            in memory representation.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getResultAsFileSystem(
            SlangInt    entryPointIndex,
            SlangInt    targetIndex, 
            ISlangMutableFileSystem** outFileSystem) = 0;

            /** Compute a hash for the entry point at `entryPointIndex` for the chosen `targetIndex`.

            This computes a hash based on all the dependencies for this component type as well as the
            target settings affecting the compiler backend. The computed hash is used as a key for caching
            the output of the compiler backend to implement shader caching.
            */
        virtual SLANG_NO_THROW void SLANG_MCALL getEntryPointHash(
            SlangInt    entryPointIndex,
            SlangInt    targetIndex,
            IBlob**     outHash) = 0;

            /** Specialize the component by binding its specialization parameters to concrete arguments.

            The `specializationArgs` array must have `specializationArgCount` entries, and
            this must match the number of specialization parameters on this component type.

            If any diagnostics (error or warnings) are produced, they will be written to `outDiagnostics`.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
            SpecializationArg const*    specializationArgs,
            SlangInt                    specializationArgCount,
            IComponentType**            outSpecializedComponentType,
            ISlangBlob**                outDiagnostics = nullptr) = 0;

            /** Link this component type against all of its unsatisifed dependencies.
            
            A component type may have unsatisfied dependencies. For example, a module
            depends on any other modules it `import`s, and an entry point depends
            on the module that defined it.

            A user can manually satisfy dependencies by creating a composite
            component type, and when doing so they retain full control over
            the relative ordering of shader parameters in the resulting layout.

            It is an error to try to generate/access compiled kernel code for
            a component type with unresolved dependencies, so if dependencies
            remain after whatever manual composition steps an application
            cares to peform, the `link()` function can be used to automatically
            compose in any remaining dependencies. The order of parameters
            (and hence the global layout) that results will be deterministic,
            but is not currently documented.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL link(
            IComponentType**            outLinkedComponentType,
            ISlangBlob**                outDiagnostics = nullptr) = 0;

            /** Get entry point 'callable' functions accessible through the ISlangSharedLibrary interface.

            The functions remain in scope as long as the ISlangSharedLibrary interface is in scope.

            NOTE! Requires a compilation target of SLANG_HOST_CALLABLE.
    
            @param entryPointIndex  The index of the entry point to get code for.
            @param targetIndex      The index of the target to get code for (default: zero).
            @param outSharedLibrary A pointer to a ISharedLibrary interface which functions can be queried on.
            @returns                A `SlangResult` to indicate success or failure.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
            int                     entryPointIndex,
            int                     targetIndex,
            ISlangSharedLibrary**   outSharedLibrary,
            slang::IBlob**          outDiagnostics = 0) = 0;

            /** Get a new ComponentType object that represents a renamed entry point.

            The current object must be a single EntryPoint, or a CompositeComponentType or
            SpecializedComponentType that contains one EntryPoint component.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL renameEntryPoint(
            const char* newName, IComponentType** outEntryPoint) = 0;
        
            /** Link and specify additional compiler options when generating code
            *   from the linked program.
            */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL linkWithOptions(
            IComponentType** outLinkedComponentType,
            uint32_t compilerOptionEntryCount,
            CompilerOptionEntry* compilerOptionEntries,
            ISlangBlob** outDiagnostics = nullptr) = 0;

        virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTargetCode(
            SlangInt targetIndex,
            IBlob** outCode,
            IBlob** outDiagnostics = nullptr) = 0;
    };
    #define SLANG_UUID_IComponentType IComponentType::getTypeGuid()

    struct IEntryPoint : public IComponentType
    {
        SLANG_COM_INTERFACE(0x8f241361, 0xf5bd, 0x4ca0, { 0xa3, 0xac, 0x2, 0xf7, 0xfa, 0x24, 0x2, 0xb8 })

        virtual SLANG_NO_THROW FunctionReflection* SLANG_MCALL getFunctionReflection() = 0;
    };

    #define SLANG_UUID_IEntryPoint IEntryPoint::getTypeGuid()

    struct ITypeConformance : public IComponentType
    {
        SLANG_COM_INTERFACE(0x73eb3147, 0xe544, 0x41b5, { 0xb8, 0xf0, 0xa2, 0x44, 0xdf, 0x21, 0x94, 0xb })
    };
    #define SLANG_UUID_ITypeConformance ITypeConformance::getTypeGuid()

        /** A module is the granularity of shader code compilation and loading.

        In most cases a module corresponds to a single compile "translation unit."
        This will often be a single `.slang` or `.hlsl` file and everything it
        `#include`s.

        Notably, a module `M` does *not* include the things it `import`s, as these
        as distinct modules that `M` depends on. There is a directed graph of
        module dependencies, and all modules in the graph must belong to the
        same session (`ISession`).

        A module establishes a namespace for looking up types, functions, etc.
        */
    struct IModule : public IComponentType
    {
        SLANG_COM_INTERFACE(0xc720e64, 0x8722, 0x4d31, { 0x89, 0x90, 0x63, 0x8a, 0x98, 0xb1, 0xc2, 0x79 })

        virtual SLANG_NO_THROW SlangResult SLANG_MCALL findEntryPointByName(
            char const*     name,
            IEntryPoint**   outEntryPoint) = 0;

        /// Get number of entry points defined in the module. An entry point defined in a module
        /// is by default not included in the linkage, so calls to `IComponentType::getEntryPointCount`
        /// on an `IModule` instance will always return 0. However `IModule::getDefinedEntryPointCount`
        /// will return the number of defined entry points.
        virtual SLANG_NO_THROW SlangInt32 SLANG_MCALL getDefinedEntryPointCount() = 0;
        /// Get the name of an entry point defined in the module.
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
            getDefinedEntryPoint(SlangInt32 index, IEntryPoint** outEntryPoint) = 0;

        /// Get a serialized representation of the checked module.
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL serialize(ISlangBlob** outSerializedBlob) = 0;

        /// Write the serialized representation of this module to a file.
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL writeToFile(char const* fileName) = 0;

        /// Get the name of the module.
        virtual SLANG_NO_THROW const char* SLANG_MCALL getName() = 0;

        /// Get the path of the module.
        virtual SLANG_NO_THROW const char* SLANG_MCALL getFilePath() = 0;

        /// Get the unique identity of the module.
        virtual SLANG_NO_THROW const char* SLANG_MCALL getUniqueIdentity() = 0;

        /// Find and validate an entry point by name, even if the function is
        /// not marked with the `[shader("...")]` attribute.
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL findAndCheckEntryPoint(
            char const* name,
            SlangStage stage,
            IEntryPoint** outEntryPoint,
            ISlangBlob** outDiagnostics) = 0;

        /// Get the number of dependency files that this module depends on.
        /// This includes both the explicit source files, as well as any
        /// additional files that were transitively referenced (e.g., via
        /// a `#include` directive).
        virtual SLANG_NO_THROW SlangInt32 SLANG_MCALL getDependencyFileCount() = 0;

        /// Get the path to a file this module depends on.
        virtual SLANG_NO_THROW char const* SLANG_MCALL getDependencyFilePath(
            SlangInt32 index) = 0;
    };
    
    #define SLANG_UUID_IModule IModule::getTypeGuid()

        /** Argument used for specialization to types/values.
        */
    struct SpecializationArg
    {
        enum class Kind : int32_t
        {
            Unknown,    /**< An invalid specialization argument. */
            Type,       /**< Specialize to a type. */
        };

        /** The kind of specialization argument. */
        Kind kind;
        union
        {
            /** A type specialization argument, used for `Kind::Type`. */
            TypeReflection* type;
        };

        static SpecializationArg fromType(TypeReflection* inType)
        {
            SpecializationArg rs;
            rs.kind = Kind::Type;
            rs.type = inType;
            return rs;
        }
    };
}

// Passed into functions to create globalSession to identify the API version client code is
// using. 
#define SLANG_API_VERSION 0

/* Create a global session, with built in StdLib.

@param apiVersion Pass in SLANG_API_VERSION
@param outGlobalSession (out)The created global session. 
*/
SLANG_EXTERN_C SLANG_API SlangResult slang_createGlobalSession(
    SlangInt                apiVersion,
    slang::IGlobalSession** outGlobalSession);

/* Create a global session, but do not set up the stdlib. The stdlib can
then be loaded via loadStdLib or compileStdLib

@param apiVersion Pass in SLANG_API_VERSION
@param outGlobalSession (out)The created global session that doesn't have a StdLib setup.

NOTE! API is experimental and not ready for production code 
*/
SLANG_EXTERN_C SLANG_API SlangResult slang_createGlobalSessionWithoutStdLib(
    SlangInt                apiVersion,
    slang::IGlobalSession** outGlobalSession);

/* Returns a blob that contains the serialized stdlib.
Returns nullptr if there isn't an embedded stdlib.
*/
SLANG_API ISlangBlob* slang_getEmbeddedStdLib();

namespace slang
{
    inline SlangResult createGlobalSession(
        slang::IGlobalSession** outGlobalSession)
    {
        return slang_createGlobalSession(SLANG_API_VERSION, outGlobalSession);
    }
}

/** @see slang::ICompileRequest::getProgram
*/
SLANG_EXTERN_C SLANG_API SlangResult spCompileRequest_getProgram(
    SlangCompileRequest*    request,
    slang::IComponentType** outProgram);

/** @see slang::ICompileRequest::getProgramWithEntryPoints
*/
SLANG_EXTERN_C SLANG_API SlangResult spCompileRequest_getProgramWithEntryPoints(
    SlangCompileRequest*    request,
    slang::IComponentType** outProgram);

/** @see slang::ICompileRequest::getEntryPoint
*/
SLANG_EXTERN_C SLANG_API SlangResult spCompileRequest_getEntryPoint(
    SlangCompileRequest*    request,
    SlangInt                entryPointIndex,
    slang::IComponentType** outEntryPoint);

/** @see slang::ICompileRequest::getModule
*/
SLANG_EXTERN_C SLANG_API SlangResult spCompileRequest_getModule(
    SlangCompileRequest*    request,
    SlangInt                translationUnitIndex,
    slang::IModule**        outModule);

/** @see slang::ICompileRequest::getSession
*/
SLANG_EXTERN_C SLANG_API SlangResult spCompileRequest_getSession(
    SlangCompileRequest* request,
    slang::ISession** outSession);
#endif

/* DEPRECATED DEFINITIONS

Everything below this point represents deprecated APIs/definition that are only
being kept around for source/binary compatibility with old client code. New
code should not use any of these declarations, and the Slang API will drop these
declarations over time.
*/

#ifdef __cplusplus
extern "C" {
#endif

#define SLANG_ERROR_INSUFFICIENT_BUFFER SLANG_E_BUFFER_TOO_SMALL
#define SLANG_ERROR_INVALID_PARAMETER SLANG_E_INVALID_ARG

SLANG_API char const* spGetTranslationUnitSource(
    SlangCompileRequest*    request,
    int                     translationUnitIndex);

#ifdef __cplusplus
}
#endif

#endif


#include <assert.h>

#include <stdint.h>

#ifndef SLANG_CORE_SIGNAL_H
#define SLANG_CORE_SIGNAL_H


namespace Slang
{

enum class SignalType
{
    Unexpected,
    Unimplemented,
    AssertFailure,
    Unreachable,
    InvalidOperation,
    AbortCompilation,
};


// Note that message can be passed as nullptr for no message.
SLANG_RETURN_NEVER void handleSignal(SignalType type, char const* message);

#define SLANG_UNEXPECTED(reason) \
    ::Slang::handleSignal(::Slang::SignalType::Unexpected, reason)

#define SLANG_UNIMPLEMENTED_X(what) \
    ::Slang::handleSignal(::Slang::SignalType::Unimplemented, what)

#define SLANG_UNREACHABLE(msg) \
    ::Slang::handleSignal(::Slang::SignalType::Unreachable, msg)

#define SLANG_ASSERT_FAILURE(msg) \
    ::Slang::handleSignal(::Slang::SignalType::AssertFailure, msg)

#define SLANG_INVALID_OPERATION(msg) \
    ::Slang::handleSignal(::Slang::SignalType::InvalidOperation, msg)

#define SLANG_ABORT_COMPILATION(msg) \
    ::Slang::handleSignal(::Slang::SignalType::AbortCompilation, msg)


}

#endif


#define VARIADIC_TEMPLATE

namespace Slang
{
    
	typedef int32_t Int32;
	typedef uint32_t UInt32;

	typedef int64_t Int64;
	typedef uint64_t UInt64;

    // Define 
    typedef SlangUInt UInt;
    typedef SlangInt Int;

    static const UInt kMaxUInt = ~UInt(0);
    static const Int kMaxInt = Int(kMaxUInt >> 1);

//	typedef unsigned short Word;

	typedef intptr_t PtrInt;

    // TODO(JS): It looks like Index is actually 64 bit on 64 bit targets(!)
    // Previous discussions landed on Index being int32_t.

    // Type used for indexing, in arrays/views etc. Signed.
    typedef Int Index;
    typedef UInt UIndex;
    typedef Int Count;
    typedef UInt UCount;

    static const Index kMaxIndex = kMaxInt;

    typedef uint8_t Byte;

    // TODO(JS):
    // Perhaps these should be named Utf8, Utf16 and UnicodePoint/Rune/etc? For now, just keep it simple
    //
    typedef char Char8;
    // 16 bit character. Note much like in utf8, a character may or may not represent a code point (it can be part of a code point).  
    typedef uint16_t Char16;

    // Can always hold a unicode code point.
    typedef uint32_t Char32;

	template <typename T>
	inline T&& _Move(T & obj)
	{
		return static_cast<T&&>(obj);
	}

	template <typename T>
	inline void Swap(T & v0, T & v1)
	{
		T tmp = _Move(v0);
		v0 = _Move(v1);
		v1 = _Move(tmp);
	}

    // Make these interfaces have more convenient names
    typedef ISlangCastable ICastable;
    typedef ISlangClonable IClonable;

    // Convenience function for using clonable
    template <typename T>
    SLANG_FORCE_INLINE T* clone(IClonable* clonable) { return (T*)clonable->clone(T::getTypeGuid()); }

    template <typename T>
    inline bool isBitSet(T value, T bitToTest)
    {
        static_assert(sizeof(T) <= sizeof(uint32_t), "Only support up to 32 bit enums");
        return (T)((uint32_t)value & (uint32_t)bitToTest) == bitToTest;
    }
}

// SLANG_DEFER
template<typename F>
class SlangDeferImpl
{
    F f;
public:
    SlangDeferImpl(F&& f)
        : f(Slang::_Move(f))
    {}
    ~SlangDeferImpl()
    {
        f();
    }
};

#ifndef SLANG_DEFER_LAMBDA
#define SLANG_DEFER_LAMBDA(x) auto SLANG_CONCAT(slang_defer_, __LINE__) = SlangDeferImpl(x)
#define SLANG_DEFER(x) auto SLANG_CONCAT(slang_defer_,__LINE__) = SlangDeferImpl([&](){x;})
#endif

//
// Some macros for avoiding boilerplate
// TODO: could probably deduce the size with templates, and move the whole
// thing into a template
//
#if __cplusplus >= 202002L
#define SLANG_COMPONENTWISE_EQUALITY_1(type) bool operator==(const type& other) const = default;
#define SLANG_COMPONENTWISE_EQUALITY_2(type) bool operator==(const type& other) const = default;
#define SLANG_COMPONENTWISE_EQUALITY_3(type) bool operator==(const type& other) const = default;
#else
#define SLANG_COMPONENTWISE_EQUALITY_1(type) \
    bool operator==(const type& other) const \
    { \
        const auto& [m1] = *this; \
        const auto& [o1] = other; \
        return m1 == o1; \
    } \
    bool operator!=(const type& other) const \
    { \
        return !(*this == other); \
    }

#define SLANG_COMPONENTWISE_EQUALITY_2(type) \
    bool operator==(const type& other) const \
    { \
        const auto& [m1, m2] = *this; \
        const auto& [o1, o2] = other; \
        return m1 == o1 && m2 == o2; \
    } \
    bool operator!=(const type& other) const \
    { \
        return !(*this == other); \
    }

#define SLANG_COMPONENTWISE_EQUALITY_3(type) \
    bool operator==(const type& other) const \
    { \
        const auto& [m1, m2, m3] = *this; \
        const auto& [o1, o2, o3] = other; \
        return m1 == o1 && m2 == o2 && m3 == o3; \
    } \
    bool operator!=(const type& other) const \
    { \
        return !(*this == other); \
    }
#endif

// TODO: Shouldn't these be SLANG_ prefixed?
#ifdef _MSC_VER
#define UNREACHABLE_RETURN(x)
#else
#define UNREACHABLE_RETURN(x) return x;
#endif

//
// Use `SLANG_ASSUME(myBoolExpression);` to inform the compiler that the condition is true.
// Do not rely on side effects of the condition being performed.
//
#if defined(__cpp_assume)
#    define SLANG_ASSUME(X) [[assume(X)]]
#elif SLANG_GCC
#    define SLANG_ASSUME(X) do{if(!(X)) __builtin_unreachable();} while(0)
#elif SLANG_CLANG
#    define SLANG_ASSUME(X) __builtin_assume(X)
#elif SLANG_VC
#    define SLANG_ASSUME(X) __assume(X)
#else
     [[noreturn]] inline void invokeUndefinedBehaviour() {}
#    define SLANG_ASSUME(X) do{if(!(X)) invokeUndefinedBehaviour();} while(0)
#endif

//
// Assertions abort in debug builds, but inform the compiler of true
// assumptions in release builds
//
#ifdef _DEBUG
#define SLANG_ASSERT(VALUE) do{if(!(VALUE)) SLANG_ASSERT_FAILURE(#VALUE);} while(0)
#else
#define SLANG_ASSERT(VALUE) SLANG_ASSUME(VALUE)
#endif

#define SLANG_RELEASE_ASSERT(VALUE) if(VALUE) {} else SLANG_ASSERT_FAILURE(#VALUE)

template<typename T> void slang_use_obj(T&) {}

#define SLANG_UNREFERENCED_PARAMETER(P) slang_use_obj(P)
#define SLANG_UNREFERENCED_VARIABLE(P) slang_use_obj(P)
#endif

#if defined(SLANG_RT_DYNAMIC)
#if defined(_MSC_VER)
#    ifdef SLANG_RT_DYNAMIC_EXPORT
#        define SLANG_RT_API SLANG_DLL_EXPORT
#    else
#        define SLANG_RT_API __declspec(dllimport)
#    endif
#else
// TODO: need to consider compiler capabilities
//#     ifdef SLANG_RT_DYNAMIC_EXPORT
#    define SLANG_RT_API SLANG_DLL_EXPORT
//#     endif
#endif
#endif

#if defined(_MSC_VER)
#   define SLANG_ATTR_PRINTF(string_index, varargs_index)
#else
#   define SLANG_ATTR_PRINTF(string_index, varargs_index) __attribute__((format(printf, string_index, varargs_index)))
#endif

#ifndef SLANG_RT_API
#define SLANG_RT_API
#endif

#ifndef SLANG_CORE_HASH_H
#define SLANG_CORE_HASH_H

#ifndef SLANG_CORE_MATH_H
#define SLANG_CORE_MATH_H

#include <cmath>

namespace Slang
{
    // Some handy constants

    // The largest positive (or negative) number 
#   define SLANG_HALF_MAX 65504.0f
    // Smallest (denormalized) value. 1 / 2^24
#   define SLANG_HALF_SUB_NORMAL_MIN (1.0f / 16777216.0f)

	class Math
	{
	public:
        // Use to fix type punning issues with strict aliasing
        union FloatIntUnion
        {
            float fvalue;
            int ivalue;

            SLANG_FORCE_INLINE static FloatIntUnion makeFromInt(int i) { FloatIntUnion cast; cast.ivalue = i; return cast; }
            SLANG_FORCE_INLINE static FloatIntUnion makeFromFloat(float f) { FloatIntUnion cast; cast.fvalue = f; return cast; }
        };
        union DoubleInt64Union
        {
            double dvalue;
            int64_t ivalue;
            SLANG_FORCE_INLINE static DoubleInt64Union makeFromInt64(int64_t i) { DoubleInt64Union cast; cast.ivalue = i; return cast; }
            SLANG_FORCE_INLINE static DoubleInt64Union makeFromDouble(double d) { DoubleInt64Union cast; cast.dvalue = d; return cast; }
        };
        
		static const float Pi;

        template <typename T>
        static T Abs(T a)
        {
            return (a < 0) ? -a : a;
        }

		template<typename T>
		static T Min(const T& v1, const T&v2)
		{
			return v1<v2?v1:v2;
		}
		template<typename T>
		static T Max(const T& v1, const T&v2)
		{
			return v1>v2?v1:v2;
		}
		template<typename T>
		static T Min(const T& v1, const T&v2, const T&v3)
		{
			return Min(v1, Min(v2, v3));
		}
		template<typename T>
		static T Max(const T& v1, const T&v2, const T&v3)
		{
			return Max(v1, Max(v2, v3));
		}
		template<typename T>
		static T Clamp(const T& val, const T& vmin, const T&vmax)
		{
			if (val < vmin) return vmin;
			else if (val > vmax) return vmax;
			else return val;
		}

		static inline int FastFloor(float x)
		{
			int i = (int)x;
			return i - (i > x);
		}

		static inline int FastFloor(double x)
		{
			int i = (int)x;
			return i - (i > x);
		}

		static inline int IsNaN(float x)
		{
			return std::isnan(x);
		}

		static inline int IsInf(float x)
		{
			return std::isinf(x);
		}

		static inline unsigned int Ones32(unsigned int x)
		{
			/* 32-bit recursive reduction using SWAR...
				but first step is mapping 2-bit values
				into sum of 2 1-bit values in sneaky way
			*/
			x -= ((x >> 1) & 0x55555555);
			x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
			x = (((x >> 4) + x) & 0x0f0f0f0f);
			x += (x >> 8);
			x += (x >> 16);
			return(x & 0x0000003f);
		}

		static inline unsigned int Log2Floor(unsigned int x)
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);
			x |= (x >> 16);
			return(Ones32(x >> 1));
		}

		static inline unsigned int Log2Ceil(unsigned int x)
		{
			int y = (x & (x - 1));
			y |= -y;
			y >>= (32 - 1);
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);
			x |= (x >> 16);
			return(Ones32(x >> 1) - y);
		}
		/*
		static inline int Log2(float x)
		{
			unsigned int ix = (unsigned int&)x;
			unsigned int exp = (ix >> 23) & 0xFF;
			int log2 = (unsigned int)(exp) - 127;

			return log2;
		}
		*/

        static bool AreNearlyEqual(double a, double b, double epsilon)
        {
            // If they are equal then we are done
            if (a == b)
            {
                return true;
            }

            const double absA = Abs(a);
            const double absB = Abs(b);
            const double diff = Abs(a - b);

            // https://en.wikipedia.org/wiki/Double_precision_floating-point_format
            const double minNormal = 2.2250738585072014e-308;
            // Either a or b are very close to being zero, so doing relative comparison isn't really appropriate
            if (a == 0.0 || b == 0.0 || (absA + absB < minNormal))
            {
                return diff < (epsilon * minNormal);
            }
            else
            {
                // Calculate a relative relative error
                return diff < epsilon * (absA + absB);
            }
        }

        template <typename T>
        static T getLowestBit(T val)
        {
            return val & (-val);
        }
	};
    inline int FloatAsInt(float val)
	{
        return Math::FloatIntUnion::makeFromFloat(val).ivalue; 
	}
    inline float IntAsFloat(int val)
	{
        return Math::FloatIntUnion::makeFromInt(val).fvalue; 
	}

    SLANG_FORCE_INLINE int64_t DoubleAsInt64(double val)
    {
        return Math::DoubleInt64Union::makeFromDouble(val).ivalue;
    }
    SLANG_FORCE_INLINE double Int64AsDouble(int64_t value)
    {
        return Math::DoubleInt64Union::makeFromInt64(value).dvalue;
    }

	inline unsigned short FloatToHalf(float val)
	{
        const auto x = FloatAsInt(val);
        
		unsigned short bits = (x >> 16) & 0x8000;
		unsigned short m = (x >> 12) & 0x07ff;
		unsigned int e = (x >> 23) & 0xff;
		if (e < 103)
			return bits;
		if (e > 142)
		{
			bits |= 0x7c00u;
			bits |= e == 255 && (x & 0x007fffffu);
			return bits;
		}
		if (e < 113)
		{
			m |= 0x0800u;
			bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
			return bits;
		}
		bits |= ((e - 112) << 10) | (m >> 1);
		bits += m & 1;
		return bits;
	}

	inline float HalfToFloat(unsigned short input)
	{
		static const auto magic = Math::FloatIntUnion::makeFromInt((127 + (127 - 15)) << 23);
		static const auto was_infnan = Math::FloatIntUnion::makeFromInt((127 + 16) << 23);
        Math::FloatIntUnion o;
		o.ivalue = (input & 0x7fff) << 13;     // exponent/mantissa bits
		o.fvalue *= magic.fvalue;                 // exponent adjust
		if (o.fvalue >= was_infnan.fvalue)        // make sure Inf/NaN survive
			o.ivalue |= 255 << 23;
		o.ivalue |= (input & 0x8000) << 16;    // sign bit
		return o.fvalue;
	}

	class Random
	{
	private:
		unsigned int seed;
	public:
		Random(int seed)
		{
			this->seed = seed;
		}
		int Next() // random between 0 and RandMax (currently 0x7fff)
		{
			return ((seed = ((seed << 12) + 150889L) % 714025) & 0x7fff);
		}
		int Next(int min, int max) // inclusive min, exclusive max
		{
			unsigned int a = ((seed = ((seed << 12) + 150889L) % 714025) & 0xFFFF);
			unsigned int b = ((seed = ((seed << 12) + 150889L) % 714025) & 0xFFFF);
			unsigned int r = (a << 16) + b;
			return min + r % (max - min);
		}
		float NextFloat()
		{
			return ((Next() << 15) + Next()) / ((float)(1 << 30));
		}
		float NextFloat(float valMin, float valMax)
		{
			return valMin + (valMax - valMin) * NextFloat();
		}
		static int RandMax()
		{
			return 0x7fff;
		}
	};
}

#endif 

///////////////////////// ankerl::unordered_dense::{map, set} /////////////////////////

// A fast & densely stored hashmap and hashset based on robin-hood backward shift deletion.
// Version 4.0.4
// https://github.com/martinus/unordered_dense
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2022-2023 Martin Leitner-Ankerl <martin.ankerl@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef ANKERL_UNORDERED_DENSE_H
#define ANKERL_UNORDERED_DENSE_H

// see https://semver.org/spec/v2.0.0.html
#define ANKERL_UNORDERED_DENSE_VERSION_MAJOR 4 // NOLINT(cppcoreguidelines-macro-usage) incompatible API changes
#define ANKERL_UNORDERED_DENSE_VERSION_MINOR 0 // NOLINT(cppcoreguidelines-macro-usage) backwards compatible functionality
#define ANKERL_UNORDERED_DENSE_VERSION_PATCH 4 // NOLINT(cppcoreguidelines-macro-usage) backwards compatible bug fixes

// API versioning with inline namespace, see https://www.foonathan.net/2018/11/inline-namespaces/

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ANKERL_UNORDERED_DENSE_VERSION_CONCAT1(major, minor, patch) v##major##_##minor##_##patch
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ANKERL_UNORDERED_DENSE_VERSION_CONCAT(major, minor, patch) ANKERL_UNORDERED_DENSE_VERSION_CONCAT1(major, minor, patch)
#define ANKERL_UNORDERED_DENSE_NAMESPACE   \
    ANKERL_UNORDERED_DENSE_VERSION_CONCAT( \
        ANKERL_UNORDERED_DENSE_VERSION_MAJOR, ANKERL_UNORDERED_DENSE_VERSION_MINOR, ANKERL_UNORDERED_DENSE_VERSION_PATCH)

#if defined(_MSVC_LANG)
#    define ANKERL_UNORDERED_DENSE_CPP_VERSION _MSVC_LANG
#else
#    define ANKERL_UNORDERED_DENSE_CPP_VERSION __cplusplus
#endif

#if defined(__GNUC__)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define ANKERL_UNORDERED_DENSE_PACK(decl) decl __attribute__((__packed__))
#elif defined(_MSC_VER)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define ANKERL_UNORDERED_DENSE_PACK(decl) __pragma(pack(push, 1)) decl __pragma(pack(pop))
#endif

// exceptions
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#    define ANKERL_UNORDERED_DENSE_HAS_EXCEPTIONS() 1 // NOLINT(cppcoreguidelines-macro-usage)
#else
#    define ANKERL_UNORDERED_DENSE_HAS_EXCEPTIONS() 0 // NOLINT(cppcoreguidelines-macro-usage)
#endif
#ifdef _MSC_VER
#    define ANKERL_UNORDERED_DENSE_NOINLINE __declspec(noinline)
#else
#    define ANKERL_UNORDERED_DENSE_NOINLINE __attribute__((noinline))
#endif

#if ANKERL_UNORDERED_DENSE_CPP_VERSION < 201703L
#    error ankerl::unordered_dense requires C++17 or higher
#else
#    include <array>            // for array
#    include <cstdint>          // for uint64_t, uint32_t, uint8_t, UINT64_C
#    include <cstring>          // for size_t, memcpy, memset
#    include <functional>       // for equal_to, hash
#    include <initializer_list> // for initializer_list
#    include <iterator>         // for pair, distance
#    include <limits>           // for numeric_limits
#    include <memory>           // for allocator, allocator_traits, shared_ptr
#    include <stdexcept>        // for out_of_range
#    include <string>           // for basic_string
#    include <string_view>      // for basic_string_view, hash
#    include <tuple>            // for forward_as_tuple
#    include <type_traits>      // for enable_if_t, declval, conditional_t, ena...
#    include <utility>          // for forward, exchange, pair, as_const, piece...
#    include <vector>           // for vector
#    if ANKERL_UNORDERED_DENSE_HAS_EXCEPTIONS() == 0
#        include <cstdlib> // for abort
#    endif

#    if defined(__has_include)
#        if __has_include(<memory_resource>)
#            define ANKERL_UNORDERED_DENSE_PMR std::pmr // NOLINT(cppcoreguidelines-macro-usage)
#            include <memory_resource>                  // for polymorphic_allocator
#        elif __has_include(<experimental/memory_resource>)
#            define ANKERL_UNORDERED_DENSE_PMR std::experimental::pmr // NOLINT(cppcoreguidelines-macro-usage)
#            include <experimental/memory_resource>                   // for polymorphic_allocator
#        endif
#    endif

#    if defined(_MSC_VER) && defined(_M_X64)
#        include <intrin.h>
#        pragma intrinsic(_umul128)
#    endif

#    if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#        define ANKERL_UNORDERED_DENSE_LIKELY(x) __builtin_expect(x, 1)   // NOLINT(cppcoreguidelines-macro-usage)
#        define ANKERL_UNORDERED_DENSE_UNLIKELY(x) __builtin_expect(x, 0) // NOLINT(cppcoreguidelines-macro-usage)
#    else
#        define ANKERL_UNORDERED_DENSE_LIKELY(x) (x)   // NOLINT(cppcoreguidelines-macro-usage)
#        define ANKERL_UNORDERED_DENSE_UNLIKELY(x) (x) // NOLINT(cppcoreguidelines-macro-usage)
#    endif

namespace ankerl::unordered_dense {
inline namespace ANKERL_UNORDERED_DENSE_NAMESPACE {

namespace detail {

#    if ANKERL_UNORDERED_DENSE_HAS_EXCEPTIONS()

// make sure this is not inlined as it is slow and dramatically enlarges code, thus making other
// inlinings more difficult. Throws are also generally the slow path.
[[noreturn]] inline ANKERL_UNORDERED_DENSE_NOINLINE void on_error_key_not_found() {
    throw std::out_of_range("ankerl::unordered_dense::map::at(): key not found");
}
[[noreturn]] inline ANKERL_UNORDERED_DENSE_NOINLINE void on_error_bucket_overflow() {
    throw std::overflow_error("ankerl::unordered_dense: reached max bucket size, cannot increase size");
}
[[noreturn]] inline ANKERL_UNORDERED_DENSE_NOINLINE void on_error_too_many_elements() {
    throw std::out_of_range("ankerl::unordered_dense::map::replace(): too many elements");
}

#    else

[[noreturn]] inline void on_error_key_not_found() {
    abort();
}
[[noreturn]] inline void on_error_bucket_overflow() {
    abort();
}
[[noreturn]] inline void on_error_too_many_elements() {
    abort();
}

#    endif

} // namespace detail

// hash ///////////////////////////////////////////////////////////////////////

// This is a stripped-down implementation of wyhash: https://github.com/wangyi-fudan/wyhash
// No big-endian support (because different values on different machines don't matter),
// hardcodes seed and the secret, reformattes the code, and clang-tidy fixes.
namespace detail::wyhash {

static inline void mum(uint64_t* a, uint64_t* b) {
#    if defined(__SIZEOF_INT128__)
    __uint128_t r = *a;
    r *= *b;
    *a = static_cast<uint64_t>(r);
    *b = static_cast<uint64_t>(r >> 64U);
#    elif defined(_MSC_VER) && defined(_M_X64)
    *a = _umul128(*a, *b, b);
#    else
    uint64_t ha = *a >> 32U;
    uint64_t hb = *b >> 32U;
    uint64_t la = static_cast<uint32_t>(*a);
    uint64_t lb = static_cast<uint32_t>(*b);
    uint64_t hi{};
    uint64_t lo{};
    uint64_t rh = ha * hb;
    uint64_t rm0 = ha * lb;
    uint64_t rm1 = hb * la;
    uint64_t rl = la * lb;
    uint64_t t = rl + (rm0 << 32U);
    auto c = static_cast<uint64_t>(t < rl);
    lo = t + (rm1 << 32U);
    c += static_cast<uint64_t>(lo < t);
    hi = rh + (rm0 >> 32U) + (rm1 >> 32U) + c;
    *a = lo;
    *b = hi;
#    endif
}

// multiply and xor mix function, aka MUM
[[nodiscard]] static inline auto mix(uint64_t a, uint64_t b) -> uint64_t {
    mum(&a, &b);
    return a ^ b;
}

// read functions. WARNING: we don't care about endianness, so results are different on big endian!
[[nodiscard]] static inline auto r8(const uint8_t* p) -> uint64_t {
    uint64_t v{};
    std::memcpy(&v, p, 8U);
    return v;
}

[[nodiscard]] static inline auto r4(const uint8_t* p) -> uint64_t {
    uint32_t v{};
    std::memcpy(&v, p, 4);
    return v;
}

// reads 1, 2, or 3 bytes
[[nodiscard]] static inline auto r3(const uint8_t* p, size_t k) -> uint64_t {
    return (static_cast<uint64_t>(p[0]) << 16U) | (static_cast<uint64_t>(p[k >> 1U]) << 8U) | p[k - 1];
}

[[maybe_unused]] [[nodiscard]] static inline auto hash(void const* key, size_t len) -> uint64_t {
    static constexpr auto secret = std::array{UINT64_C(0xa0761d6478bd642f),
                                              UINT64_C(0xe7037ed1a0b428db),
                                              UINT64_C(0x8ebc6af09c88c6e3),
                                              UINT64_C(0x589965cc75374cc3)};

    auto const* p = static_cast<uint8_t const*>(key);
    uint64_t seed = secret[0];
    uint64_t a{};
    uint64_t b{};
    if (ANKERL_UNORDERED_DENSE_LIKELY(len <= 16)) {
        if (ANKERL_UNORDERED_DENSE_LIKELY(len >= 4)) {
            a = (r4(p) << 32U) | r4(p + ((len >> 3U) << 2U));
            b = (r4(p + len - 4) << 32U) | r4(p + len - 4 - ((len >> 3U) << 2U));
        } else if (ANKERL_UNORDERED_DENSE_LIKELY(len > 0)) {
            a = r3(p, len);
            b = 0;
        } else {
            a = 0;
            b = 0;
        }
    } else {
        size_t i = len;
        if (ANKERL_UNORDERED_DENSE_UNLIKELY(i > 48)) {
            uint64_t see1 = seed;
            uint64_t see2 = seed;
            do {
                seed = mix(r8(p) ^ secret[1], r8(p + 8) ^ seed);
                see1 = mix(r8(p + 16) ^ secret[2], r8(p + 24) ^ see1);
                see2 = mix(r8(p + 32) ^ secret[3], r8(p + 40) ^ see2);
                p += 48;
                i -= 48;
            } while (ANKERL_UNORDERED_DENSE_LIKELY(i > 48));
            seed ^= see1 ^ see2;
        }
        while (ANKERL_UNORDERED_DENSE_UNLIKELY(i > 16)) {
            seed = mix(r8(p) ^ secret[1], r8(p + 8) ^ seed);
            i -= 16;
            p += 16;
        }
        a = r8(p + i - 16);
        b = r8(p + i - 8);
    }

    return mix(secret[1] ^ len, mix(a ^ secret[1], b ^ seed));
}

[[nodiscard]] static inline auto hash(uint64_t x) -> uint64_t {
    return detail::wyhash::mix(x, UINT64_C(0x9E3779B97F4A7C15));
}

} // namespace detail::wyhash

template <typename T, typename Enable = void>
struct hash {
    auto operator()(T const& obj) const noexcept(noexcept(std::declval<std::hash<T>>().operator()(std::declval<T const&>())))
        -> uint64_t {
        return std::hash<T>{}(obj);
    }
};

template <typename CharT>
struct hash<std::basic_string<CharT>> {
    using is_avalanching = void;
    auto operator()(std::basic_string<CharT> const& str) const noexcept -> uint64_t {
        return detail::wyhash::hash(str.data(), sizeof(CharT) * str.size());
    }
};

template <typename CharT>
struct hash<std::basic_string_view<CharT>> {
    using is_avalanching = void;
    auto operator()(std::basic_string_view<CharT> const& sv) const noexcept -> uint64_t {
        return detail::wyhash::hash(sv.data(), sizeof(CharT) * sv.size());
    }
};

template <class T>
struct hash<T*> {
    using is_avalanching = void;
    auto operator()(T* ptr) const noexcept -> uint64_t {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return detail::wyhash::hash(reinterpret_cast<uintptr_t>(ptr));
    }
};

template <class T>
struct hash<std::unique_ptr<T>> {
    using is_avalanching = void;
    auto operator()(std::unique_ptr<T> const& ptr) const noexcept -> uint64_t {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return detail::wyhash::hash(reinterpret_cast<uintptr_t>(ptr.get()));
    }
};

template <class T>
struct hash<std::shared_ptr<T>> {
    using is_avalanching = void;
    auto operator()(std::shared_ptr<T> const& ptr) const noexcept -> uint64_t {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return detail::wyhash::hash(reinterpret_cast<uintptr_t>(ptr.get()));
    }
};

template <typename Enum>
struct hash<Enum, typename std::enable_if<std::is_enum<Enum>::value>::type> {
    using is_avalanching = void;
    auto operator()(Enum e) const noexcept -> uint64_t {
        using underlying = typename std::underlying_type_t<Enum>;
        return detail::wyhash::hash(static_cast<underlying>(e));
    }
};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define ANKERL_UNORDERED_DENSE_HASH_STATICCAST(T)                    \
        template <>                                                      \
        struct hash<T> {                                                 \
            using is_avalanching = void;                                 \
            auto operator()(T const& obj) const noexcept -> uint64_t {   \
                return detail::wyhash::hash(static_cast<uint64_t>(obj)); \
            }                                                            \
        }

#    if defined(__GNUC__) && !defined(__clang__)
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wuseless-cast"
#    endif
// see https://en.cppreference.com/w/cpp/utility/hash
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(bool);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(char);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(signed char);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(unsigned char);
#    if ANKERL_UNORDERED_DENSE_CPP_VERSION >= 202002L
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(char8_t);
#    endif
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(char16_t);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(char32_t);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(wchar_t);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(short);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(unsigned short);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(int);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(unsigned int);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(long);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(long long);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(unsigned long);
ANKERL_UNORDERED_DENSE_HASH_STATICCAST(unsigned long long);

#    if defined(__GNUC__) && !defined(__clang__)
#        pragma GCC diagnostic pop
#    endif

// bucket_type //////////////////////////////////////////////////////////

namespace bucket_type {

struct standard {
    static constexpr uint32_t dist_inc = 1U << 8U;             // skip 1 byte fingerprint
    static constexpr uint32_t fingerprint_mask = dist_inc - 1; // mask for 1 byte of fingerprint

    uint32_t m_dist_and_fingerprint; // upper 3 byte: distance to original bucket. lower byte: fingerprint from hash
    uint32_t m_value_idx;            // index into the m_values vector.
};

ANKERL_UNORDERED_DENSE_PACK(struct big {
    static constexpr uint32_t dist_inc = 1U << 8U;             // skip 1 byte fingerprint
    static constexpr uint32_t fingerprint_mask = dist_inc - 1; // mask for 1 byte of fingerprint

    uint32_t m_dist_and_fingerprint; // upper 3 byte: distance to original bucket. lower byte: fingerprint from hash
    size_t m_value_idx;              // index into the m_values vector.
});

} // namespace bucket_type

namespace detail {

struct nonesuch {};

template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<detail::nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;

template <typename T>
using detect_avalanching = typename T::is_avalanching;

template <typename T>
using detect_is_transparent = typename T::is_transparent;

template <typename T>
using detect_iterator = typename T::iterator;

template <typename T>
using detect_reserve = decltype(std::declval<T&>().reserve(size_t{}));

// enable_if helpers

template <typename Mapped>
constexpr bool is_map_v = !std::is_void_v<Mapped>;

// clang-format off
template <typename Hash, typename KeyEqual>
constexpr bool is_transparent_v = is_detected_v<detect_is_transparent, Hash> && is_detected_v<detect_is_transparent, KeyEqual>;
// clang-format on

template <typename From, typename To1, typename To2>
constexpr bool is_neither_convertible_v = !std::is_convertible_v<From, To1> && !std::is_convertible_v<From, To2>;

template <typename T>
constexpr bool has_reserve = is_detected_v<detect_reserve, T>;

// base type for map has mapped_type
template <class T>
struct base_table_type_map {
    using mapped_type = T;
};

// base type for set doesn't have mapped_type
struct base_table_type_set {};

} // namespace detail

// Very much like std::deque, but faster for indexing (in most cases). As of now this doesn't implement the full std::vector
// API, but merely what's necessary to work as an underlying container for ankerl::unordered_dense::{map, set}.
// It allocates blocks of equal size and puts them into the m_blocks vector. That means it can grow simply by adding a new
// block to the back of m_blocks, and doesn't double its size like an std::vector. The disadvantage is that memory is not
// linear and thus there is one more indirection necessary for indexing.
template <typename T, typename Allocator = std::allocator<T>, size_t MaxSegmentSizeBytes = 4096>
class segmented_vector {
    template <bool IsConst>
    class iter_t;

public:
    using allocator_type = Allocator;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
    using difference_type = typename std::allocator_traits<allocator_type>::difference_type;
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = T const&;
    using iterator = iter_t<false>;
    using const_iterator = iter_t<true>;

private:
    using vec_alloc = typename std::allocator_traits<Allocator>::template rebind_alloc<pointer>;
    std::vector<pointer, vec_alloc> m_blocks{};
    size_t m_size{};

    // Calculates the maximum number for x in  (s << x) <= max_val
    static constexpr auto num_bits_closest(size_t max_val, size_t s) -> size_t {
        auto f = size_t{0};
        while (s << (f + 1) <= max_val) {
            ++f;
        }
        return f;
    }

    using self_t = segmented_vector<T, Allocator, MaxSegmentSizeBytes>;
    static constexpr auto num_bits = num_bits_closest(MaxSegmentSizeBytes, sizeof(T));
    static constexpr auto num_elements_in_block = 1U << num_bits;
    static constexpr auto mask = num_elements_in_block - 1U;

    /**
     * Iterator class doubles as const_iterator and iterator
     */
    template <bool IsConst>
    class iter_t {
        using ptr_t = typename std::conditional_t<IsConst, segmented_vector::const_pointer const*, segmented_vector::pointer*>;
        ptr_t m_data{};
        size_t m_idx{};

        template <bool B>
        friend class iter_t;

    public:
        using difference_type = segmented_vector::difference_type;
        using value_type = T;
        using reference = typename std::conditional_t<IsConst, value_type const&, value_type&>;
        using pointer = typename std::conditional_t<IsConst, segmented_vector::const_pointer, segmented_vector::pointer>;
        using iterator_category = std::forward_iterator_tag;

        iter_t() noexcept = default;

        template <bool OtherIsConst, typename = typename std::enable_if<IsConst && !OtherIsConst>::type>
        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        constexpr iter_t(iter_t<OtherIsConst> const& other) noexcept
            : m_data(other.m_data)
            , m_idx(other.m_idx) {}

        constexpr iter_t(ptr_t data, size_t idx) noexcept
            : m_data(data)
            , m_idx(idx) {}

        template <bool OtherIsConst, typename = typename std::enable_if<IsConst && !OtherIsConst>::type>
        constexpr auto operator=(iter_t<OtherIsConst> const& other) noexcept -> iter_t& {
            m_data = other.m_data;
            m_idx = other.m_idx;
            return *this;
        }

        constexpr auto operator++() noexcept -> iter_t& {
            ++m_idx;
            return *this;
        }

        constexpr auto operator+(difference_type diff) noexcept -> iter_t {
            return {m_data, static_cast<size_t>(static_cast<difference_type>(m_idx) + diff)};
        }

        template <bool OtherIsConst>
        constexpr auto operator-(iter_t<OtherIsConst> const& other) noexcept -> difference_type {
            return static_cast<difference_type>(m_idx) - static_cast<difference_type>(other.m_idx);
        }

        constexpr auto operator*() const noexcept -> reference {
            return m_data[m_idx >> num_bits][m_idx & mask];
        }

        constexpr auto operator->() const noexcept -> pointer {
            return &m_data[m_idx >> num_bits][m_idx & mask];
        }

        template <bool O>
        constexpr auto operator==(iter_t<O> const& o) const noexcept -> bool {
            return m_idx == o.m_idx;
        }

        template <bool O>
        constexpr auto operator!=(iter_t<O> const& o) const noexcept -> bool {
            return !(*this == o);
        }
    };

    // slow path: need to allocate a new segment every once in a while
    void increase_capacity() {
        auto ba = Allocator(m_blocks.get_allocator());
        pointer block = std::allocator_traits<Allocator>::allocate(ba, num_elements_in_block);
        m_blocks.push_back(block);
    }

    // Moves everything from other
    void append_everything_from(segmented_vector&& other) {
        reserve(size() + other.size());
        for (auto&& o : other) {
            emplace_back(std::move(o));
        }
    }

    // Copies everything from other
    void append_everything_from(segmented_vector const& other) {
        reserve(size() + other.size());
        for (auto const& o : other) {
            emplace_back(o);
        }
    }

    void dealloc() {
        auto ba = Allocator(m_blocks.get_allocator());
        for (auto ptr : m_blocks) {
            std::allocator_traits<Allocator>::deallocate(ba, ptr, num_elements_in_block);
        }
    }

    [[nodiscard]] static constexpr auto calc_num_blocks_for_capacity(size_t capacity) {
        return (capacity + num_elements_in_block - 1U) / num_elements_in_block;
    }

public:
    segmented_vector() = default;

    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    segmented_vector(Allocator alloc)
        : m_blocks(vec_alloc(alloc)) {}

    segmented_vector(segmented_vector&& other, Allocator alloc)
        : segmented_vector(alloc) {
        *this = std::move(other);
    }

    segmented_vector(segmented_vector const& other, Allocator alloc)
        : m_blocks(vec_alloc(alloc)) {
        append_everything_from(other);
    }

    segmented_vector(segmented_vector&& other) noexcept
        : segmented_vector(std::move(other), get_allocator()) {}

    segmented_vector(segmented_vector const& other) {
        append_everything_from(other);
    }

    auto operator=(segmented_vector const& other) -> segmented_vector& {
        if (this == &other) {
            return *this;
        }
        clear();
        append_everything_from(other);
        return *this;
    }

    auto operator=(segmented_vector&& other) noexcept -> segmented_vector& {
        clear();
        dealloc();
        if (other.get_allocator() == get_allocator()) {
            m_blocks = std::move(other.m_blocks);
            m_size = std::exchange(other.m_size, {});
        } else {
            // make sure to construct with other's allocator!
            m_blocks = std::vector<pointer, vec_alloc>(vec_alloc(other.get_allocator()));
            append_everything_from(std::move(other));
        }
        return *this;
    }

    ~segmented_vector() {
        clear();
        dealloc();
    }

    [[nodiscard]] constexpr auto size() const -> size_t {
        return m_size;
    }

    [[nodiscard]] constexpr auto capacity() const -> size_t {
        return m_blocks.size() * num_elements_in_block;
    }

    // Indexing is highly performance critical
    [[nodiscard]] constexpr auto operator[](size_t i) const noexcept -> T const& {
        return m_blocks[i >> num_bits][i & mask];
    }

    [[nodiscard]] constexpr auto operator[](size_t i) noexcept -> T& {
        return m_blocks[i >> num_bits][i & mask];
    }

    [[nodiscard]] constexpr auto begin() -> iterator {
        return {m_blocks.data(), 0U};
    }
    [[nodiscard]] constexpr auto begin() const -> const_iterator {
        return {m_blocks.data(), 0U};
    }
    [[nodiscard]] constexpr auto cbegin() const -> const_iterator {
        return {m_blocks.data(), 0U};
    }

    [[nodiscard]] constexpr auto end() -> iterator {
        return {m_blocks.data(), m_size};
    }
    [[nodiscard]] constexpr auto end() const -> const_iterator {
        return {m_blocks.data(), m_size};
    }
    [[nodiscard]] constexpr auto cend() const -> const_iterator {
        return {m_blocks.data(), m_size};
    }

    [[nodiscard]] constexpr auto back() -> reference {
        return operator[](m_size - 1);
    }
    [[nodiscard]] constexpr auto back() const -> const_reference {
        return operator[](m_size - 1);
    }

    void pop_back() {
        back().~T();
        --m_size;
    }

    [[nodiscard]] auto empty() const {
        return 0 == m_size;
    }

    void reserve(size_t new_capacity) {
        m_blocks.reserve(calc_num_blocks_for_capacity(new_capacity));
        while (new_capacity > capacity()) {
            increase_capacity();
        }
    }

    [[nodiscard]] auto get_allocator() const -> allocator_type {
        return allocator_type{m_blocks.get_allocator()};
    }

    template <class... Args>
    auto emplace_back(Args&&... args) -> reference {
        if (m_size == capacity()) {
            increase_capacity();
        }
        auto* ptr = static_cast<void*>(&operator[](m_size));
        auto& ref = *new (ptr) T(std::forward<Args>(args)...);
        ++m_size;
        return ref;
    }

    void clear() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = 0, s = size(); i < s; ++i) {
                operator[](i).~T();
            }
        }
        m_size = 0;
    }

    void shrink_to_fit() {
        auto ba = Allocator(m_blocks.get_allocator());
        auto num_blocks_required = calc_num_blocks_for_capacity(m_size);
        while (m_blocks.size() > num_blocks_required) {
            std::allocator_traits<Allocator>::deallocate(ba, m_blocks.back(), num_elements_in_block);
            m_blocks.pop_back();
        }
        m_blocks.shrink_to_fit();
    }
};

namespace detail {

// This is it, the table. Doubles as map and set, and uses `void` for T when its used as a set.
template <class Key,
          class T, // when void, treat it as a set.
          class Hash,
          class KeyEqual,
          class AllocatorOrContainer,
          class Bucket,
          bool IsSegmented>
class table : public std::conditional_t<is_map_v<T>, base_table_type_map<T>, base_table_type_set> {
    using underlying_value_type = typename std::conditional_t<is_map_v<T>, std::pair<Key, T>, Key>;
    using underlying_container_type = std::conditional_t<IsSegmented,
                                                         segmented_vector<underlying_value_type, AllocatorOrContainer>,
                                                         std::vector<underlying_value_type, AllocatorOrContainer>>;

public:
    using value_container_type = std::
        conditional_t<is_detected_v<detect_iterator, AllocatorOrContainer>, AllocatorOrContainer, underlying_container_type>;

private:
    using bucket_alloc =
        typename std::allocator_traits<typename value_container_type::allocator_type>::template rebind_alloc<Bucket>;
    using bucket_alloc_traits = std::allocator_traits<bucket_alloc>;

    static constexpr uint8_t initial_shifts = 64 - 3; // 2^(64-m_shift) number of buckets
    static constexpr float default_max_load_factor = 0.8F;

public:
    using key_type = Key;
    using value_type = typename value_container_type::value_type;
    using size_type = typename value_container_type::size_type;
    using difference_type = typename value_container_type::difference_type;
    using hasher = Hash;
    using key_equal = KeyEqual;
    using allocator_type = typename value_container_type::allocator_type;
    using reference = typename value_container_type::reference;
    using const_reference = typename value_container_type::const_reference;
    using pointer = typename value_container_type::pointer;
    using const_pointer = typename value_container_type::const_pointer;
    using const_iterator = typename value_container_type::const_iterator;
    using iterator = std::conditional_t<is_map_v<T>, typename value_container_type::iterator, const_iterator>;
    using bucket_type = Bucket;

private:
    using value_idx_type = decltype(Bucket::m_value_idx);
    using dist_and_fingerprint_type = decltype(Bucket::m_dist_and_fingerprint);

    static_assert(std::is_trivially_destructible_v<Bucket>, "assert there's no need to call destructor / std::destroy");
    static_assert(std::is_trivially_copyable_v<Bucket>, "assert we can just memset / memcpy");

    value_container_type m_values{}; // Contains all the key-value pairs in one densely stored container. No holes.
    using bucket_pointer = typename std::allocator_traits<bucket_alloc>::pointer;
    bucket_pointer m_buckets{};
    size_t m_num_buckets = 0;
    size_t m_max_bucket_capacity = 0;
    float m_max_load_factor = default_max_load_factor;
    Hash m_hash{};
    KeyEqual m_equal{};
    uint8_t m_shifts = initial_shifts;

    [[nodiscard]] auto next(value_idx_type bucket_idx) const -> value_idx_type {
        return ANKERL_UNORDERED_DENSE_UNLIKELY(bucket_idx + 1U == m_num_buckets)
                   ? 0
                   : static_cast<value_idx_type>(bucket_idx + 1U);
    }

    // Helper to access bucket through pointer types
    [[nodiscard]] static constexpr auto at(bucket_pointer bucket_ptr, size_t offset) -> Bucket& {
        return *(bucket_ptr + static_cast<typename std::allocator_traits<bucket_alloc>::difference_type>(offset));
    }

    // use the dist_inc and dist_dec functions so that uint16_t types work without warning
    [[nodiscard]] static constexpr auto dist_inc(dist_and_fingerprint_type x) -> dist_and_fingerprint_type {
        return static_cast<dist_and_fingerprint_type>(x + Bucket::dist_inc);
    }

    [[nodiscard]] static constexpr auto dist_dec(dist_and_fingerprint_type x) -> dist_and_fingerprint_type {
        return static_cast<dist_and_fingerprint_type>(x - Bucket::dist_inc);
    }

    // The goal of mixed_hash is to always produce a high quality 64bit hash.
    template <typename K>
    [[nodiscard]] constexpr auto mixed_hash(K const& key) const -> uint64_t {
        if constexpr (is_detected_v<detect_avalanching, Hash>) {
            // we know that the hash is good because is_avalanching.
            if constexpr (sizeof(decltype(m_hash(key))) < sizeof(uint64_t)) {
                // 32bit hash and is_avalanching => multiply with a constant to avalanche bits upwards
                return m_hash(key) * UINT64_C(0x9ddfea08eb382d69);
            } else {
                // 64bit and is_avalanching => only use the hash itself.
                return m_hash(key);
            }
        } else {
            // not is_avalanching => apply wyhash
            return wyhash::hash(m_hash(key));
        }
    }

    [[nodiscard]] constexpr auto dist_and_fingerprint_from_hash(uint64_t hash) const -> dist_and_fingerprint_type {
        return Bucket::dist_inc | (static_cast<dist_and_fingerprint_type>(hash) & Bucket::fingerprint_mask);
    }

    [[nodiscard]] constexpr auto bucket_idx_from_hash(uint64_t hash) const -> value_idx_type {
        return static_cast<value_idx_type>(hash >> m_shifts);
    }

    [[nodiscard]] static constexpr auto get_key(value_type const& vt) -> key_type const& {
        if constexpr (is_map_v<T>) {
            return vt.first;
        } else {
            return vt;
        }
    }

    template <typename K>
    [[nodiscard]] auto next_while_less(K const& key) const -> Bucket {
        auto hash = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
        auto bucket_idx = bucket_idx_from_hash(hash);

        while (dist_and_fingerprint < at(m_buckets, bucket_idx).m_dist_and_fingerprint) {
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }
        return {dist_and_fingerprint, bucket_idx};
    }

    void place_and_shift_up(Bucket bucket, value_idx_type place) {
        while (0 != at(m_buckets, place).m_dist_and_fingerprint) {
            bucket = std::exchange(at(m_buckets, place), bucket);
            bucket.m_dist_and_fingerprint = dist_inc(bucket.m_dist_and_fingerprint);
            place = next(place);
        }
        at(m_buckets, place) = bucket;
    }

    [[nodiscard]] static constexpr auto calc_num_buckets(uint8_t shifts) -> size_t {
        return (std::min)(max_bucket_count(), size_t{1} << (64U - shifts));
    }

    [[nodiscard]] constexpr auto calc_shifts_for_size(size_t s) const -> uint8_t {
        auto shifts = initial_shifts;
        while (shifts > 0 && static_cast<size_t>(static_cast<float>(calc_num_buckets(shifts)) * max_load_factor()) < s) {
            --shifts;
        }
        return shifts;
    }

    // assumes m_values has data, m_buckets=m_buckets_end=nullptr, m_shifts is INITIAL_SHIFTS
    void copy_buckets(table const& other) {
        if (!empty()) {
            m_shifts = other.m_shifts;
            allocate_buckets_from_shift();
            std::memcpy(m_buckets, other.m_buckets, sizeof(Bucket) * bucket_count());
        }
    }

    /**
     * True when no element can be added any more without increasing the size
     */
    [[nodiscard]] auto is_full() const -> bool {
        return size() >= m_max_bucket_capacity;
    }

    void deallocate_buckets() {
        auto ba = bucket_alloc(m_values.get_allocator());
        if (nullptr != m_buckets) {
            bucket_alloc_traits::deallocate(ba, m_buckets, bucket_count());
            m_buckets = nullptr;
        }
        m_num_buckets = 0;
        m_max_bucket_capacity = 0;
    }

    void allocate_buckets_from_shift() {
        auto ba = bucket_alloc(m_values.get_allocator());
        m_num_buckets = calc_num_buckets(m_shifts);
        m_buckets = bucket_alloc_traits::allocate(ba, m_num_buckets);
        if (m_num_buckets == max_bucket_count()) {
            // reached the maximum, make sure we can use each bucket
            m_max_bucket_capacity = max_bucket_count();
        } else {
            m_max_bucket_capacity = static_cast<value_idx_type>(static_cast<float>(m_num_buckets) * max_load_factor());
        }
    }

    void clear_buckets() {
        if (m_buckets != nullptr) {
            std::memset(&*m_buckets, 0, sizeof(Bucket) * bucket_count());
        }
    }

    void clear_and_fill_buckets_from_values() {
        clear_buckets();
        for (value_idx_type value_idx = 0, end_idx = static_cast<value_idx_type>(m_values.size()); value_idx < end_idx;
             ++value_idx) {
            auto const& key = get_key(m_values[value_idx]);
            auto [dist_and_fingerprint, bucket] = next_while_less(key);

            // we know for certain that key has not yet been inserted, so no need to check it.
            place_and_shift_up({dist_and_fingerprint, value_idx}, bucket);
        }
    }

    void increase_size() {
        if (ANKERL_UNORDERED_DENSE_UNLIKELY(m_max_bucket_capacity == max_bucket_count())) {
            on_error_bucket_overflow();
        }
        --m_shifts;
        deallocate_buckets();
        allocate_buckets_from_shift();
        clear_and_fill_buckets_from_values();
    }

    void do_erase(value_idx_type bucket_idx) {
        auto const value_idx_to_remove = at(m_buckets, bucket_idx).m_value_idx;

        // shift down until either empty or an element with correct spot is found
        auto next_bucket_idx = next(bucket_idx);
        while (at(m_buckets, next_bucket_idx).m_dist_and_fingerprint >= Bucket::dist_inc * 2) {
            at(m_buckets, bucket_idx) = {dist_dec(at(m_buckets, next_bucket_idx).m_dist_and_fingerprint),
                                         at(m_buckets, next_bucket_idx).m_value_idx};
            bucket_idx = std::exchange(next_bucket_idx, next(next_bucket_idx));
        }
        at(m_buckets, bucket_idx) = {};

        // update m_values
        if (value_idx_to_remove != m_values.size() - 1) {
            // no luck, we'll have to replace the value with the last one and update the index accordingly
            auto& val = m_values[value_idx_to_remove];
            val = std::move(m_values.back());

            // update the values_idx of the moved entry. No need to play the info game, just look until we find the values_idx
            auto mh = mixed_hash(get_key(val));
            bucket_idx = bucket_idx_from_hash(mh);

            auto const values_idx_back = static_cast<value_idx_type>(m_values.size() - 1);
            while (values_idx_back != at(m_buckets, bucket_idx).m_value_idx) {
                bucket_idx = next(bucket_idx);
            }
            at(m_buckets, bucket_idx).m_value_idx = value_idx_to_remove;
        }
        m_values.pop_back();
    }

    template <typename K>
    auto do_erase_key(K&& key) -> size_t {
        if (empty()) {
            return 0;
        }

        auto [dist_and_fingerprint, bucket_idx] = next_while_less(key);

        while (dist_and_fingerprint == at(m_buckets, bucket_idx).m_dist_and_fingerprint &&
               !m_equal(key, get_key(m_values[at(m_buckets, bucket_idx).m_value_idx]))) {
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }

        if (dist_and_fingerprint != at(m_buckets, bucket_idx).m_dist_and_fingerprint) {
            return 0;
        }
        do_erase(bucket_idx);
        return 1;
    }

    template <class K, class M>
    auto do_insert_or_assign(K&& key, M&& mapped) -> std::pair<iterator, bool> {
        auto it_isinserted = try_emplace(std::forward<K>(key), std::forward<M>(mapped));
        if (!it_isinserted.second) {
            it_isinserted.first->second = std::forward<M>(mapped);
        }
        return it_isinserted;
    }

    template <typename K, typename... Args>
    auto do_place_element(dist_and_fingerprint_type dist_and_fingerprint, value_idx_type bucket_idx, K&& key, Args&&... args)
        -> std::pair<iterator, bool> {

        // emplace the new value. If that throws an exception, no harm done; index is still in a valid state
        m_values.emplace_back(std::piecewise_construct,
                              std::forward_as_tuple(std::forward<K>(key)),
                              std::forward_as_tuple(std::forward<Args>(args)...));

        // place element and shift up until we find an empty spot
        auto value_idx = static_cast<value_idx_type>(m_values.size() - 1);
        place_and_shift_up({dist_and_fingerprint, value_idx}, bucket_idx);
        return {begin() + static_cast<difference_type>(value_idx), true};
    }

    template <typename K, typename... Args>
    auto do_try_emplace(K&& key, Args&&... args) -> std::pair<iterator, bool> {
        if (ANKERL_UNORDERED_DENSE_UNLIKELY(is_full())) {
            increase_size();
        }

        auto hash = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
        auto bucket_idx = bucket_idx_from_hash(hash);

        while (true) {
            auto* bucket = &at(m_buckets, bucket_idx);
            if (dist_and_fingerprint == bucket->m_dist_and_fingerprint) {
                if (m_equal(key, m_values[bucket->m_value_idx].first)) {
                    return {begin() + static_cast<difference_type>(bucket->m_value_idx), false};
                }
            } else if (dist_and_fingerprint > bucket->m_dist_and_fingerprint) {
                return do_place_element(dist_and_fingerprint, bucket_idx, std::forward<K>(key), std::forward<Args>(args)...);
            }
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }
    }

    template <typename K>
    auto do_find(K const& key) -> iterator {
        if (ANKERL_UNORDERED_DENSE_UNLIKELY(empty())) {
            return end();
        }

        auto mh = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(mh);
        auto bucket_idx = bucket_idx_from_hash(mh);
        auto* bucket = &at(m_buckets, bucket_idx);

        // unrolled loop. *Always* check a few directly, then enter the loop. This is faster.
        if (dist_and_fingerprint == bucket->m_dist_and_fingerprint && m_equal(key, get_key(m_values[bucket->m_value_idx]))) {
            return begin() + static_cast<difference_type>(bucket->m_value_idx);
        }
        dist_and_fingerprint = dist_inc(dist_and_fingerprint);
        bucket_idx = next(bucket_idx);
        bucket = &at(m_buckets, bucket_idx);

        if (dist_and_fingerprint == bucket->m_dist_and_fingerprint && m_equal(key, get_key(m_values[bucket->m_value_idx]))) {
            return begin() + static_cast<difference_type>(bucket->m_value_idx);
        }
        dist_and_fingerprint = dist_inc(dist_and_fingerprint);
        bucket_idx = next(bucket_idx);
        bucket = &at(m_buckets, bucket_idx);

        while (true) {
            if (dist_and_fingerprint == bucket->m_dist_and_fingerprint) {
                if (m_equal(key, get_key(m_values[bucket->m_value_idx]))) {
                    return begin() + static_cast<difference_type>(bucket->m_value_idx);
                }
            } else if (dist_and_fingerprint > bucket->m_dist_and_fingerprint) {
                return end();
            }
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
            bucket = &at(m_buckets, bucket_idx);
        }
    }

    template <typename K>
    auto do_find(K const& key) const -> const_iterator {
        return const_cast<table*>(this)->do_find(key); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    }

    template <typename K, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto do_at(K const& key) -> Q& {
        if (auto it = find(key); ANKERL_UNORDERED_DENSE_LIKELY(end() != it)) {
            return it->second;
        }
        on_error_key_not_found();
    }

    template <typename K, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto do_at(K const& key) const -> Q const& {
        return const_cast<table*>(this)->at(key); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    }

public:
    table()
        : table(0) {}

    explicit table(size_t bucket_count,
                   Hash const& hash = Hash(),
                   KeyEqual const& equal = KeyEqual(),
                   allocator_type const& alloc_or_container = allocator_type())
        : m_values(alloc_or_container)
        , m_hash(hash)
        , m_equal(equal) {
        if (0 != bucket_count) {
            reserve(bucket_count);
        }
    }

    table(size_t bucket_count, allocator_type const& alloc)
        : table(bucket_count, Hash(), KeyEqual(), alloc) {}

    table(size_t bucket_count, Hash const& hash, allocator_type const& alloc)
        : table(bucket_count, hash, KeyEqual(), alloc) {}

    explicit table(allocator_type const& alloc)
        : table(0, Hash(), KeyEqual(), alloc) {}

    template <class InputIt>
    table(InputIt first,
          InputIt last,
          size_type bucket_count = 0,
          Hash const& hash = Hash(),
          KeyEqual const& equal = KeyEqual(),
          allocator_type const& alloc = allocator_type())
        : table(bucket_count, hash, equal, alloc) {
        insert(first, last);
    }

    template <class InputIt>
    table(InputIt first, InputIt last, size_type bucket_count, allocator_type const& alloc)
        : table(first, last, bucket_count, Hash(), KeyEqual(), alloc) {}

    template <class InputIt>
    table(InputIt first, InputIt last, size_type bucket_count, Hash const& hash, allocator_type const& alloc)
        : table(first, last, bucket_count, hash, KeyEqual(), alloc) {}

    table(table const& other)
        : table(other, other.m_values.get_allocator()) {}

    table(table const& other, allocator_type const& alloc)
        : m_values(other.m_values, alloc)
        , m_max_load_factor(other.m_max_load_factor)
        , m_hash(other.m_hash)
        , m_equal(other.m_equal) {
        copy_buckets(other);
    }

    table(table&& other) noexcept
        : table(std::move(other), other.m_values.get_allocator()) {}

    table(table&& other, allocator_type const& alloc) noexcept
        : m_values(alloc) {
        *this = std::move(other);
    }

    table(std::initializer_list<value_type> ilist,
          size_t bucket_count = 0,
          Hash const& hash = Hash(),
          KeyEqual const& equal = KeyEqual(),
          allocator_type const& alloc = allocator_type())
        : table(bucket_count, hash, equal, alloc) {
        insert(ilist);
    }

    table(std::initializer_list<value_type> ilist, size_type bucket_count, allocator_type const& alloc)
        : table(ilist, bucket_count, Hash(), KeyEqual(), alloc) {}

    table(std::initializer_list<value_type> init, size_type bucket_count, Hash const& hash, allocator_type const& alloc)
        : table(init, bucket_count, hash, KeyEqual(), alloc) {}

    ~table() {
        if (nullptr != m_buckets) {
            auto ba = bucket_alloc(m_values.get_allocator());
            bucket_alloc_traits::deallocate(ba, m_buckets, bucket_count());
        }
    }

    auto operator=(table const& other) -> table& {
        if (&other != this) {
            deallocate_buckets(); // deallocate before m_values is set (might have another allocator)
            m_values = other.m_values;
            m_max_load_factor = other.m_max_load_factor;
            m_hash = other.m_hash;
            m_equal = other.m_equal;
            m_shifts = initial_shifts;
            copy_buckets(other);
        }
        return *this;
    }

    auto operator=(table&& other) noexcept(
        noexcept(std::is_nothrow_move_assignable_v<value_container_type>&& std::is_nothrow_move_assignable_v<Hash>&&
                     std::is_nothrow_move_assignable_v<KeyEqual>)) -> table& {
        if (&other != this) {
            deallocate_buckets(); // deallocate before m_values is set (might have another allocator)
            m_values = std::move(other.m_values);
            other.m_values.clear();

            // we can only reuse m_buckets when both maps have the same allocator!
            if (get_allocator() == other.get_allocator()) {
                m_buckets = std::exchange(other.m_buckets, nullptr);
                m_num_buckets = std::exchange(other.m_num_buckets, 0);
                m_max_bucket_capacity = std::exchange(other.m_max_bucket_capacity, 0);
                m_shifts = std::exchange(other.m_shifts, initial_shifts);
                m_max_load_factor = std::exchange(other.m_max_load_factor, default_max_load_factor);
                m_hash = std::exchange(other.m_hash, {});
                m_equal = std::exchange(other.m_equal, {});
            } else {
                // set max_load_factor *before* copying the other's buckets, so we have the same
                // behavior
                m_max_load_factor = other.m_max_load_factor;

                // copy_buckets sets m_buckets, m_num_buckets, m_max_bucket_capacity, m_shifts
                copy_buckets(other);
                // clear's the other's buckets so other is now already usable.
                other.clear_buckets();
                m_hash = other.m_hash;
                m_equal = other.m_equal;
            }
            // map "other" is now already usable, it's empty.
        }
        return *this;
    }

    auto operator=(std::initializer_list<value_type> ilist) -> table& {
        clear();
        insert(ilist);
        return *this;
    }

    auto get_allocator() const noexcept -> allocator_type {
        return m_values.get_allocator();
    }

    // iterators //////////////////////////////////////////////////////////////

    auto begin() noexcept -> iterator {
        return m_values.begin();
    }

    auto begin() const noexcept -> const_iterator {
        return m_values.begin();
    }

    auto cbegin() const noexcept -> const_iterator {
        return m_values.cbegin();
    }

    auto end() noexcept -> iterator {
        return m_values.end();
    }

    auto cend() const noexcept -> const_iterator {
        return m_values.cend();
    }

    auto end() const noexcept -> const_iterator {
        return m_values.end();
    }

    // capacity ///////////////////////////////////////////////////////////////

    [[nodiscard]] auto empty() const noexcept -> bool {
        return m_values.empty();
    }

    [[nodiscard]] auto size() const noexcept -> size_t {
        return m_values.size();
    }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_t {
        if constexpr ((std::numeric_limits<value_idx_type>::max)() == (std::numeric_limits<size_t>::max)()) {
            return size_t{1} << (sizeof(value_idx_type) * 8 - 1);
        } else {
            return size_t{1} << (sizeof(value_idx_type) * 8);
        }
    }

    // modifiers //////////////////////////////////////////////////////////////

    void clear() {
        m_values.clear();
        clear_buckets();
    }

    auto insert(value_type const& value) -> std::pair<iterator, bool> {
        return emplace(value);
    }

    auto insert(value_type&& value) -> std::pair<iterator, bool> {
        return emplace(std::move(value));
    }

    template <class P, std::enable_if_t<std::is_constructible_v<value_type, P&&>, bool> = true>
    auto insert(P&& value) -> std::pair<iterator, bool> {
        return emplace(std::forward<P>(value));
    }

    auto insert(const_iterator /*hint*/, value_type const& value) -> iterator {
        return insert(value).first;
    }

    auto insert(const_iterator /*hint*/, value_type&& value) -> iterator {
        return insert(std::move(value)).first;
    }

    template <class P, std::enable_if_t<std::is_constructible_v<value_type, P&&>, bool> = true>
    auto insert(const_iterator /*hint*/, P&& value) -> iterator {
        return insert(std::forward<P>(value)).first;
    }

    template <class InputIt>
    void insert(InputIt first, InputIt last) {
        while (first != last) {
            insert(*first);
            ++first;
        }
    }

    void insert(std::initializer_list<value_type> ilist) {
        insert(ilist.begin(), ilist.end());
    }

    // nonstandard API: *this is emptied.
    // Also see "A Standard flat_map" https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0429r9.pdf
    auto extract() && -> value_container_type {
        return std::move(m_values);
    }

    // nonstandard API:
    // Discards the internally held container and replaces it with the one passed. Erases non-unique elements.
    auto replace(value_container_type&& container) {
        if (ANKERL_UNORDERED_DENSE_UNLIKELY(container.size() > max_size())) {
            on_error_too_many_elements();
        }
        auto shifts = calc_shifts_for_size(container.size());
        if (0 == m_num_buckets || shifts < m_shifts || container.get_allocator() != m_values.get_allocator()) {
            m_shifts = shifts;
            deallocate_buckets();
            allocate_buckets_from_shift();
        }
        clear_buckets();

        m_values = std::move(container);

        // can't use clear_and_fill_buckets_from_values() because container elements might not be unique
        auto value_idx = value_idx_type{};

        // loop until we reach the end of the container. duplicated entries will be replaced with back().
        while (value_idx != static_cast<value_idx_type>(m_values.size())) {
            auto const& key = get_key(m_values[value_idx]);

            auto hash = mixed_hash(key);
            auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
            auto bucket_idx = bucket_idx_from_hash(hash);

            bool key_found = false;
            while (true) {
                auto const& bucket = at(m_buckets, bucket_idx);
                if (dist_and_fingerprint > bucket.m_dist_and_fingerprint) {
                    break;
                }
                if (dist_and_fingerprint == bucket.m_dist_and_fingerprint &&
                    m_equal(key, m_values[bucket.m_value_idx].first)) {
                    key_found = true;
                    break;
                }
                dist_and_fingerprint = dist_inc(dist_and_fingerprint);
                bucket_idx = next(bucket_idx);
            }

            if (key_found) {
                if (value_idx != static_cast<value_idx_type>(m_values.size() - 1)) {
                    m_values[value_idx] = std::move(m_values.back());
                }
                m_values.pop_back();
            } else {
                place_and_shift_up({dist_and_fingerprint, value_idx}, bucket_idx);
                ++value_idx;
            }
        }
    }

    template <class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(Key const& key, M&& mapped) -> std::pair<iterator, bool> {
        return do_insert_or_assign(key, std::forward<M>(mapped));
    }

    template <class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(Key&& key, M&& mapped) -> std::pair<iterator, bool> {
        return do_insert_or_assign(std::move(key), std::forward<M>(mapped));
    }

    template <typename K,
              typename M,
              typename Q = T,
              typename H = Hash,
              typename KE = KeyEqual,
              std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto insert_or_assign(K&& key, M&& mapped) -> std::pair<iterator, bool> {
        return do_insert_or_assign(std::forward<K>(key), std::forward<M>(mapped));
    }

    template <class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, Key const& key, M&& mapped) -> iterator {
        return do_insert_or_assign(key, std::forward<M>(mapped)).first;
    }

    template <class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, Key&& key, M&& mapped) -> iterator {
        return do_insert_or_assign(std::move(key), std::forward<M>(mapped)).first;
    }

    template <typename K,
              typename M,
              typename Q = T,
              typename H = Hash,
              typename KE = KeyEqual,
              std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, K&& key, M&& mapped) -> iterator {
        return do_insert_or_assign(std::forward<K>(key), std::forward<M>(mapped)).first;
    }

    // Single arguments for unordered_set can be used without having to construct the value_type
    template <class K,
              typename Q = T,
              typename H = Hash,
              typename KE = KeyEqual,
              std::enable_if_t<!is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto emplace(K&& key) -> std::pair<iterator, bool> {
        if (is_full()) {
            increase_size();
        }

        auto hash = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
        auto bucket_idx = bucket_idx_from_hash(hash);

        while (dist_and_fingerprint <= at(m_buckets, bucket_idx).m_dist_and_fingerprint) {
            if (dist_and_fingerprint == at(m_buckets, bucket_idx).m_dist_and_fingerprint &&
                m_equal(key, m_values[at(m_buckets, bucket_idx).m_value_idx])) {
                // found it, return without ever actually creating anything
                return {begin() + static_cast<difference_type>(at(m_buckets, bucket_idx).m_value_idx), false};
            }
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }

        // value is new, insert element first, so when exception happens we are in a valid state
        m_values.emplace_back(std::forward<K>(key));
        // now place the bucket and shift up until we find an empty spot
        auto value_idx = static_cast<value_idx_type>(m_values.size() - 1);
        place_and_shift_up({dist_and_fingerprint, value_idx}, bucket_idx);
        return {begin() + static_cast<difference_type>(value_idx), true};
    }

    template <class... Args>
    auto emplace(Args&&... args) -> std::pair<iterator, bool> {
        if (is_full()) {
            increase_size();
        }

        // we have to instantiate the value_type to be able to access the key.
        // 1. emplace_back the object so it is constructed. 2. If the key is already there, pop it later in the loop.
        auto& key = get_key(m_values.emplace_back(std::forward<Args>(args)...));
        auto hash = mixed_hash(key);
        auto dist_and_fingerprint = dist_and_fingerprint_from_hash(hash);
        auto bucket_idx = bucket_idx_from_hash(hash);

        while (dist_and_fingerprint <= at(m_buckets, bucket_idx).m_dist_and_fingerprint) {
            if (dist_and_fingerprint == at(m_buckets, bucket_idx).m_dist_and_fingerprint &&
                m_equal(key, get_key(m_values[at(m_buckets, bucket_idx).m_value_idx]))) {
                m_values.pop_back(); // value was already there, so get rid of it
                return {begin() + static_cast<difference_type>(at(m_buckets, bucket_idx).m_value_idx), false};
            }
            dist_and_fingerprint = dist_inc(dist_and_fingerprint);
            bucket_idx = next(bucket_idx);
        }

        // value is new, place the bucket and shift up until we find an empty spot
        auto value_idx = static_cast<value_idx_type>(m_values.size() - 1);
        place_and_shift_up({dist_and_fingerprint, value_idx}, bucket_idx);

        return {begin() + static_cast<difference_type>(value_idx), true};
    }

    template <class... Args>
    auto emplace_hint(const_iterator /*hint*/, Args&&... args) -> iterator {
        return emplace(std::forward<Args>(args)...).first;
    }

    template <class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(Key const& key, Args&&... args) -> std::pair<iterator, bool> {
        return do_try_emplace(key, std::forward<Args>(args)...);
    }

    template <class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(Key&& key, Args&&... args) -> std::pair<iterator, bool> {
        return do_try_emplace(std::move(key), std::forward<Args>(args)...);
    }

    template <class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(const_iterator /*hint*/, Key const& key, Args&&... args) -> iterator {
        return do_try_emplace(key, std::forward<Args>(args)...).first;
    }

    template <class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(const_iterator /*hint*/, Key&& key, Args&&... args) -> iterator {
        return do_try_emplace(std::move(key), std::forward<Args>(args)...).first;
    }

    template <
        typename K,
        typename... Args,
        typename Q = T,
        typename H = Hash,
        typename KE = KeyEqual,
        std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE> && is_neither_convertible_v<K&&, iterator, const_iterator>,
                         bool> = true>
    auto try_emplace(K&& key, Args&&... args) -> std::pair<iterator, bool> {
        return do_try_emplace(std::forward<K>(key), std::forward<Args>(args)...);
    }

    template <
        typename K,
        typename... Args,
        typename Q = T,
        typename H = Hash,
        typename KE = KeyEqual,
        std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE> && is_neither_convertible_v<K&&, iterator, const_iterator>,
                         bool> = true>
    auto try_emplace(const_iterator /*hint*/, K&& key, Args&&... args) -> iterator {
        return do_try_emplace(std::forward<K>(key), std::forward<Args>(args)...).first;
    }

    auto erase(iterator it) -> iterator {
        auto hash = mixed_hash(get_key(*it));
        auto bucket_idx = bucket_idx_from_hash(hash);

        auto const value_idx_to_remove = static_cast<value_idx_type>(it - cbegin());
        while (at(m_buckets, bucket_idx).m_value_idx != value_idx_to_remove) {
            bucket_idx = next(bucket_idx);
        }

        do_erase(bucket_idx);
        return begin() + static_cast<difference_type>(value_idx_to_remove);
    }

    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto erase(const_iterator it) -> iterator {
        return erase(begin() + (it - cbegin()));
    }

    auto erase(const_iterator first, const_iterator last) -> iterator {
        auto const idx_first = first - cbegin();
        auto const idx_last = last - cbegin();
        auto const first_to_last = std::distance(first, last);
        auto const last_to_end = std::distance(last, cend());

        // remove elements from left to right which moves elements from the end back
        auto const mid = idx_first + (std::min)(first_to_last, last_to_end);
        auto idx = idx_first;
        while (idx != mid) {
            erase(begin() + idx);
            ++idx;
        }

        // all elements from the right are moved, now remove the last element until all done
        idx = idx_last;
        while (idx != mid) {
            --idx;
            erase(begin() + idx);
        }

        return begin() + idx_first;
    }

    auto erase(Key const& key) -> size_t {
        return do_erase_key(key);
    }

    template <class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto erase(K&& key) -> size_t {
        return do_erase_key(std::forward<K>(key));
    }

    void swap(table& other) noexcept(noexcept(std::is_nothrow_swappable_v<value_container_type>&&
                                                  std::is_nothrow_swappable_v<Hash>&& std::is_nothrow_swappable_v<KeyEqual>)) {
        using std::swap;
        swap(other, *this);
    }

    // lookup /////////////////////////////////////////////////////////////////

    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto at(key_type const& key) -> Q& {
        return do_at(key);
    }

    template <typename K,
              typename Q = T,
              typename H = Hash,
              typename KE = KeyEqual,
              std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto at(K const& key) -> Q& {
        return do_at(key);
    }

    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto at(key_type const& key) const -> Q const& {
        return do_at(key);
    }

    template <typename K,
              typename Q = T,
              typename H = Hash,
              typename KE = KeyEqual,
              std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto at(K const& key) const -> Q const& {
        return do_at(key);
    }

    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto operator[](Key const& key) -> Q& {
        return try_emplace(key).first->second;
    }

    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto operator[](Key&& key) -> Q& {
        return try_emplace(std::move(key)).first->second;
    }

    template <typename K,
              typename Q = T,
              typename H = Hash,
              typename KE = KeyEqual,
              std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto operator[](K&& key) -> Q& {
        return try_emplace(std::forward<K>(key)).first->second;
    }

    auto count(Key const& key) const -> size_t {
        return find(key) == end() ? 0 : 1;
    }

    template <class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto count(K const& key) const -> size_t {
        return find(key) == end() ? 0 : 1;
    }

    auto find(Key const& key) -> iterator {
        return do_find(key);
    }

    auto find(Key const& key) const -> const_iterator {
        return do_find(key);
    }

    template <class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto find(K const& key) -> iterator {
        return do_find(key);
    }

    template <class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto find(K const& key) const -> const_iterator {
        return do_find(key);
    }

    auto contains(Key const& key) const -> bool {
        return find(key) != end();
    }

    template <class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto contains(K const& key) const -> bool {
        return find(key) != end();
    }

    auto equal_range(Key const& key) -> std::pair<iterator, iterator> {
        auto it = do_find(key);
        return {it, it == end() ? end() : it + 1};
    }

    auto equal_range(const Key& key) const -> std::pair<const_iterator, const_iterator> {
        auto it = do_find(key);
        return {it, it == end() ? end() : it + 1};
    }

    template <class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto equal_range(K const& key) -> std::pair<iterator, iterator> {
        auto it = do_find(key);
        return {it, it == end() ? end() : it + 1};
    }

    template <class K, class H = Hash, class KE = KeyEqual, std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto equal_range(K const& key) const -> std::pair<const_iterator, const_iterator> {
        auto it = do_find(key);
        return {it, it == end() ? end() : it + 1};
    }

    // bucket interface ///////////////////////////////////////////////////////

    auto bucket_count() const noexcept -> size_t { // NOLINT(modernize-use-nodiscard)
        return m_num_buckets;
    }

    static constexpr auto max_bucket_count() noexcept -> size_t { // NOLINT(modernize-use-nodiscard)
        return max_size();
    }

    // hash policy ////////////////////////////////////////////////////////////

    [[nodiscard]] auto load_factor() const -> float {
        return bucket_count() ? static_cast<float>(size()) / static_cast<float>(bucket_count()) : 0.0F;
    }

    [[nodiscard]] auto max_load_factor() const -> float {
        return m_max_load_factor;
    }

    void max_load_factor(float ml) {
        m_max_load_factor = ml;
        if (m_num_buckets != max_bucket_count()) {
            m_max_bucket_capacity = static_cast<value_idx_type>(static_cast<float>(bucket_count()) * max_load_factor());
        }
    }

    void rehash(size_t count) {
        count = (std::min)(count, max_size());
        auto shifts = calc_shifts_for_size((std::max)(count, size()));
        if (shifts != m_shifts) {
            m_shifts = shifts;
            deallocate_buckets();
            m_values.shrink_to_fit();
            allocate_buckets_from_shift();
            clear_and_fill_buckets_from_values();
        }
    }

    void reserve(size_t capa) {
        capa = (std::min)(capa, max_size());
        if constexpr (has_reserve<value_container_type>) {
            // std::deque doesn't have reserve(). Make sure we only call when available
            m_values.reserve(capa);
        }
        auto shifts = calc_shifts_for_size((std::max)(capa, size()));
        if (0 == m_num_buckets || shifts < m_shifts) {
            m_shifts = shifts;
            deallocate_buckets();
            allocate_buckets_from_shift();
            clear_and_fill_buckets_from_values();
        }
    }

    // observers //////////////////////////////////////////////////////////////

    auto hash_function() const -> hasher {
        return m_hash;
    }

    auto key_eq() const -> key_equal {
        return m_equal;
    }

    // nonstandard API: expose the underlying values container
    [[nodiscard]] auto values() const noexcept -> value_container_type const& {
        return m_values;
    }

    // non-member functions ///////////////////////////////////////////////////

    friend auto operator==(table const& a, table const& b) -> bool {
        if (&a == &b) {
            return true;
        }
        if (a.size() != b.size()) {
            return false;
        }
        for (auto const& b_entry : b) {
            auto it = a.find(get_key(b_entry));
            if constexpr (is_map_v<T>) {
                // map: check that key is here, then also check that value is the same
                if (a.end() == it || !(b_entry.second == it->second)) {
                    return false;
                }
            } else {
                // set: only check that the key is here
                if (a.end() == it) {
                    return false;
                }
            }
        }
        return true;
    }

    friend auto operator!=(table const& a, table const& b) -> bool {
        return !(a == b);
    }
};

} // namespace detail

template <class Key,
          class T,
          class Hash = hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
          class Bucket = bucket_type::standard>
using map = detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, false>;

template <class Key,
          class T,
          class Hash = hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
          class Bucket = bucket_type::standard>
using segmented_map = detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, true>;

template <class Key,
          class Hash = hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<Key>,
          class Bucket = bucket_type::standard>
using set = detail::table<Key, void, Hash, KeyEqual, AllocatorOrContainer, Bucket, false>;

template <class Key,
          class Hash = hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<Key>,
          class Bucket = bucket_type::standard>
using segmented_set = detail::table<Key, void, Hash, KeyEqual, AllocatorOrContainer, Bucket, true>;

#    if defined(ANKERL_UNORDERED_DENSE_PMR)

namespace pmr {

template <class Key,
          class T,
          class Hash = hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class Bucket = bucket_type::standard>
using map =
    detail::table<Key, T, Hash, KeyEqual, ANKERL_UNORDERED_DENSE_PMR::polymorphic_allocator<std::pair<Key, T>>, Bucket, false>;

template <class Key,
          class T,
          class Hash = hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class Bucket = bucket_type::standard>
using segmented_map =
    detail::table<Key, T, Hash, KeyEqual, ANKERL_UNORDERED_DENSE_PMR::polymorphic_allocator<std::pair<Key, T>>, Bucket, true>;

template <class Key, class Hash = hash<Key>, class KeyEqual = std::equal_to<Key>, class Bucket = bucket_type::standard>
using set = detail::table<Key, void, Hash, KeyEqual, ANKERL_UNORDERED_DENSE_PMR::polymorphic_allocator<Key>, Bucket, false>;

template <class Key, class Hash = hash<Key>, class KeyEqual = std::equal_to<Key>, class Bucket = bucket_type::standard>
using segmented_set =
    detail::table<Key, void, Hash, KeyEqual, ANKERL_UNORDERED_DENSE_PMR::polymorphic_allocator<Key>, Bucket, true>;

} // namespace pmr

#    endif

// deduction guides ///////////////////////////////////////////////////////////

// deduction guides for alias templates are only possible since C++20
// see https://en.cppreference.com/w/cpp/language/class_template_argument_deduction

} // namespace ANKERL_UNORDERED_DENSE_NAMESPACE
} // namespace ankerl::unordered_dense

// std extensions /////////////////////////////////////////////////////////////

namespace std { // NOLINT(cert-dcl58-cpp)

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          class AllocatorOrContainer,
          class Bucket,
          class Pred,
          bool IsSegmented>
// NOLINTNEXTLINE(cert-dcl58-cpp)
auto erase_if(ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, IsSegmented>& map,
              Pred pred) -> size_t {
    using map_t = ankerl::unordered_dense::detail::table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, IsSegmented>;

    // going back to front because erase() invalidates the end iterator
    auto const old_size = map.size();
    auto idx = old_size;
    while (idx) {
        --idx;
        auto it = map.begin() + static_cast<typename map_t::difference_type>(idx);
        if (pred(*it)) {
            map.erase(it);
        }
    }

    return old_size - map.size();
}

} // namespace std

#endif
#endif

#include <cstring>
#include <type_traits>

namespace Slang
{
    //
    // Types
    //

    // A fixed 64bit wide hash on all targets.
    typedef uint64_t HashCode64;
    typedef HashCode64 HashCode;
    // A fixed 32bit wide hash on all targets.
    typedef uint32_t HashCode32;

    //
    // Some helpers to determine which hash to use for a type
    //

    // Forward declare Hash
    template<typename T> struct Hash;

    template<typename T, typename = void>
    constexpr static bool HasSlangHash = false;
    template<typename T>
    constexpr static bool HasSlangHash<
        T,
        std::enable_if_t<std::is_convertible_v<
            decltype((std::declval<const T&>()).getHashCode()),
            HashCode64>>>
        = true;

    // Does the hashmap implementation provide a uniform hash for this type.
    template<typename T, typename = void>
    constexpr static bool HasWyhash = false;
    template<typename T>
    constexpr static bool HasWyhash<T, typename ankerl::unordered_dense::hash<T>::is_avalanching> = true;

    // We want to have an associated type 'is_avalanching = void' iff we have a
    // hash with good uniformity, the two specializations here add that member
    // when appropriate (since we can't declare an associated type with
    // constexpr if or something terse like that)
    template <typename T, typename = void>
    struct DetectAvalanchingHash {};
    template <typename T>
    struct DetectAvalanchingHash<T, std::enable_if_t<HasWyhash<T>>>
    {
        using is_avalanching = void;
    };
    // Have we marked 'getHashCode' as having good uniformity properties.
    template <typename T>
    struct DetectAvalanchingHash<T, std::enable_if_t<T::kHasUniformHash>>
    {
        using is_avalanching = void;
    };

    // A helper for hashing according to the bit representation
    template<typename T, typename U>
    struct BitCastHash : DetectAvalanchingHash<U>
    {
        auto operator()(const T& t) const
        {
            // Doesn't discard or invent bits
            static_assert(sizeof(T) == sizeof(U));
            // Can we copy bytes to and fro
            static_assert(std::is_trivially_copyable_v<T>);
            static_assert(std::is_trivially_copyable_v<U>);
            // Because we construct a U to memcpy into
            static_assert(std::is_trivially_constructible_v<U>);

            U u;
            memcpy(&u, &t, sizeof(T));
            return Hash<U>{}(u);
        }
    };

    //
    // Our hashing functor which disptaches to the most appropriate hashing
    // function for the type
    //

    template<typename T>
    struct Hash : DetectAvalanchingHash<T>
    {
        auto operator()(const T& t) const
        {
            // Our preference is for any hash we've defined ourselves
            if constexpr (HasSlangHash<T>)
                return t.getHashCode();
            // Otherwise fall back to any good hash provided by the hashmap
            // library
            else if constexpr (HasWyhash<T>)
                return ankerl::unordered_dense::hash<T>{}(t);
            // Otherwise fail
            else
            {
                // !sizeof(T*) is a 'false' which is dependent on T (pending P2593R0)
                static_assert(!sizeof(T*), "No hash implementation found for this type");
                // This is to avoid the return type being deduced as 'void' and creating further errors.
                return HashCode64(0);
            }
        }
    };

    // Specializations for float and double which hash 0 and -0 to distinct values
    template<>
    struct Hash<float> : BitCastHash<float, uint32_t> {};
    template<>
    struct Hash<double> : BitCastHash<double, uint64_t> {};

    //
    // Utility functions for using hashes
    //

    // A wrapper for Hash<TKey>
	template<typename TKey>
	auto getHashCode(const TKey& key)
	{
        return Hash<TKey>{}(key);
	}

	inline HashCode64 getHashCode(const char* buffer, std::size_t len)
	{
        return ankerl::unordered_dense::detail::wyhash::hash(buffer, len);
	}

    template<typename T>
    HashCode64 hashObjectBytes(const T& t)
    {
        static_assert(std::has_unique_object_representations_v<T>,
            "This type must have a unique object representation to use hashObjectBytes");
        return getHashCode(reinterpret_cast<const char*>(&t), sizeof(t));
    }

    // Use in a struct to declare a uniform hash which doens't care about the
    // structure of the members.
#   define SLANG_BYTEWISE_HASHABLE \
        static constexpr bool kHasUniformHash = true; \
        ::Slang::HashCode64 getHashCode() const \
        { \
            return ::Slang::hashObjectBytes(*this); \
        }

#   define SLANG_COMPONENTWISE_HASHABLE_1 \
        auto getHashCode() const \
        { \
            const auto& [m1] = *this; \
            return Slang::getHashCode(m1); \
        }

#   define SLANG_COMPONENTWISE_HASHABLE_2 \
        auto getHashCode() const \
        { \
            const auto& [m1, m2] = *this; \
            return combineHash(::Slang::getHashCode(m1), ::Slang::getHashCode(m2)); \
        }

    inline HashCode64 combineHash(HashCode64 h)
    {
        return h;
    }

    inline HashCode32 combineHash(HashCode32 h)
    {
        return h;
    }

    // A left fold of a mixing operation
    template<typename H1, typename H2, typename... Hs>
    auto combineHash(H1 n, H2 m, Hs... args)
    {
        // TODO: restrict the types here more, currently we tend to throw
        // unhashed integers in here along with proper hashes of objects.
        static_assert(std::is_convertible_v<H1, HashCode64> || std::is_convertible_v<H1, HashCode32>);
        static_assert(std::is_convertible_v<H2, HashCode64> || std::is_convertible_v<H2, HashCode32>);
        return combineHash((n * 16777619) ^ m, args...);
    }

    struct Hasher
    {
    public:
        Hasher() {}

            /// Hash the given `value` and combine it into this hash state
        template<typename T>
        void hashValue(T const& value)
        {
            // TODO: Eventually, we should replace `getHashCode`
            // with a "hash into" operation that takes the value
            // and a `Hasher`.

            m_hashCode = combineHash(m_hashCode, getHashCode(value));
        }

            /// Combine the given `hash` code into the hash state.
            ///
            /// Note: users should prefer to use `hashValue` or `hashObject`
            /// when possible, as they may be able to ensure a higher-quality
            /// hash result (e.g., by using more bits to represent the state
            /// during hashing than are used for the final hash code).
            ///
        void addHash(HashCode hash)
        {
            m_hashCode = combineHash(m_hashCode, hash);
        }

        HashCode getResult() const
        {
            return m_hashCode;
        }

    private:
        HashCode m_hashCode = 0;
    };
}

#endif

#ifndef SLANG_CORE_TYPE_TRAITS_H
#define SLANG_CORE_TYPE_TRAITS_H

namespace Slang
{
	struct TraitResultYes
	{
		char x;
	};
	struct TraitResultNo
	{
		char x[2];
	};

	template <typename B, typename D>
	struct IsBaseOfTraitHost
	{
		operator B*() const { return nullptr; }
		operator D*() { return nullptr; }
	};

	template <typename B, typename D>
	struct IsBaseOf
	{
		template <typename T>
		static TraitResultYes Check(D*, T) { return TraitResultYes(); }
		static TraitResultNo Check(B*, int) { return TraitResultNo(); }
		enum { Value = sizeof(Check(IsBaseOfTraitHost<B, D>(), int())) == sizeof(TraitResultYes) };
	};

	template<bool B, class T = void>
	struct EnableIf {};

	template<class T>
	struct EnableIf<true, T> { typedef T type; };

	template <typename B, typename D>
	struct IsConvertible
	{
		static TraitResultYes Use(B) { return TraitResultYes(); };
		static TraitResultNo Use(...) { return TraitResultNo(); };
		enum { Value = sizeof(Use(*(D*)(nullptr))) == sizeof(TraitResultYes) };
	};
}

#endif



namespace Slang
{
    // Base class for all reference-counted objects
    class SLANG_RT_API RefObject
    {
    private:
        UInt referenceCount;

    public:
        RefObject()
            : referenceCount(0)
        {}

        RefObject(const RefObject &)
            : referenceCount(0)
        {}

        RefObject& operator=(const RefObject&) { return *this; }

        virtual ~RefObject()
        {}

        UInt addReference()
        {
            return ++referenceCount;
        }

        UInt decreaseReference()
        {
            return --referenceCount;
        }

        UInt releaseReference()
        {
            SLANG_ASSERT(referenceCount != 0);
            if(--referenceCount == 0)
            {
                delete this;
                return 0;
            }
            return referenceCount;
        }

        bool isUniquelyReferenced()
        {
            SLANG_ASSERT(referenceCount != 0);
            return referenceCount == 1;
        }

        UInt debugGetReferenceCount()
        {
            return referenceCount;
        }
    };

    SLANG_FORCE_INLINE void addReference(RefObject* obj)
    {
        if(obj) obj->addReference();
    }

    SLANG_FORCE_INLINE void releaseReference(RefObject* obj)
    {
        if(obj) obj->releaseReference();
    }

    // For straight dynamic cast.
    // Use instead of dynamic_cast as it allows for replacement without using Rtti in the future
    template <typename T>
    SLANG_FORCE_INLINE T* dynamicCast(RefObject* obj) { return dynamic_cast<T*>(obj); }
    template <typename T>
    SLANG_FORCE_INLINE const T* dynamicCast(const RefObject* obj) { return dynamic_cast<const T*>(obj); }

    // Like a dynamicCast, but allows a type to implement a specific implementation that is suitable for it
    template <typename T>
    SLANG_FORCE_INLINE T* as(RefObject* obj) { return dynamicCast<T>(obj); }
    template <typename T>
    SLANG_FORCE_INLINE const T* as(const RefObject* obj) { return dynamicCast<T>(obj); }

    // "Smart" pointer to a reference-counted object
    template<typename T> struct SLANG_RT_API RefPtr
    {
        RefPtr()
            : pointer(nullptr)
        {}

        RefPtr(T* p)
            : pointer(p)
        {
            addReference(p);
        }

        RefPtr(RefPtr<T> const& p)
            : pointer(p.pointer)
        {
            addReference(p.pointer);
        }

        RefPtr(RefPtr<T>&& p)
            : pointer(p.pointer)
        {
            p.pointer = nullptr;
        }

        template <typename U>
        RefPtr(RefPtr<U> const& p,
            typename EnableIf<IsConvertible<T*, U*>::Value, void>::type * = 0)
            : pointer(static_cast<U*>(p))
        {
            addReference(static_cast<U*>(p));
        }

#if 0
        void operator=(T* p)
        {
            T* old = pointer;
            addReference(p);
            pointer = p;
            releaseReference(old);
        }
#endif

        void operator=(RefPtr<T> const& p)
        {
            T* old = pointer;
            addReference(p.pointer);
            pointer = p.pointer;
            releaseReference(old);
        }

        void operator=(RefPtr<T>&& p)
        {
            T* old = pointer;
            pointer = p.pointer;
            p.pointer = old;
        }

        template <typename U>
        typename EnableIf<IsConvertible<T*, U*>::value, void>::type
            operator=(RefPtr<U> const& p)
        {
            T* old = pointer;
            addReference(p.pointer);
            pointer = p.pointer;
            releaseReference(old);
        }

        HashCode getHashCode() const
        {
            // Note: We need a `RefPtr<T>` to hash the same as a `T*`,
            // so that a `T*` can be used as a key in a dictionary with
            // `RefPtr<T>` keys, and vice versa.
            //
            return Slang::getHashCode(pointer);
        }

        bool operator==(const T * ptr) const
        {
            return pointer == ptr;
        }

        bool operator!=(const T * ptr) const
        {
            return pointer != ptr;
        }

		bool operator==(RefPtr<T> const& ptr) const
		{
			return pointer == ptr.pointer;
		}

		bool operator!=(RefPtr<T> const& ptr) const
		{
			return pointer != ptr.pointer;
		}

        template<typename U>
        RefPtr<U> dynamicCast() const
        {
            return RefPtr<U>(Slang::dynamicCast<U>(pointer));
        }

        template<typename U>
        RefPtr<U> as() const
        {
            return RefPtr<U>(Slang::as<U>(pointer));
        }

        template <typename U>
        bool is() const { return Slang::as<U>(pointer) != nullptr; }

        ~RefPtr()
        {
            releaseReference(static_cast<Slang::RefObject*>(pointer));
        }

        T& operator*() const
        {
            return *pointer;
        }

        T* operator->() const
        {
            return pointer;
        }

		T * Ptr() const
		{
			return pointer;
		}

        T* get() const
        {
            return pointer;
        }

        operator T*() const
        {
            return pointer;
        }

        void attach(T* p)
        {
            T* old = pointer;
            pointer = p;
            releaseReference(old);
        }

        T* detach()
        {
            auto rs = pointer;
            pointer = nullptr;
            return rs;
        }

        void swapWith(RefPtr<T>& rhs)
        {
            auto rhsPtr = rhs.pointer;
            rhs.pointer = pointer;
            pointer = rhsPtr;
        }

        SLANG_FORCE_INLINE void setNull()
        {
            releaseReference(pointer);
            pointer = nullptr;
        }

        /// Get ready for writing (nulls contents)
        SLANG_FORCE_INLINE T** writeRef() { *this = nullptr; return &pointer; }

        /// Get for read access
        SLANG_FORCE_INLINE T*const* readRef() const { return &pointer; }

    private:
        T* pointer;
	};

    // Helper type for implementing weak pointers. The object being pointed at weakly creates a WeakSink object
    // that other objects can reference and share. When the object is destroyed it detaches the sink
    // doing so will make other users call to 'get' return null. Thus any user of the WeakSink, must check if the weakly pointed to
    // things pointer is nullptr before using.
    template <typename T>
    class WeakSink : public RefObject
    {
    public:
        WeakSink(T* ptr):
            m_ptr(ptr)
        {
        }

        SLANG_FORCE_INLINE T* get() const { return m_ptr; }
        SLANG_FORCE_INLINE void detach() { m_ptr = nullptr; }

    private:
        T* m_ptr;
    };

    // A pointer that can be transformed to hold either a weak reference or a strong reference.
    template<typename T>
    class TransformablePtr
    {
    private:
        T* m_weakPtr = nullptr;
        RefPtr<T> m_strongPtr;

    public:
        TransformablePtr() = default;
        TransformablePtr(T* ptr) { *this = ptr; }
        TransformablePtr(RefPtr<T> ptr) { *this = ptr; }
        TransformablePtr(const TransformablePtr<T>& ptr) = default;
        TransformablePtr<T>& operator=(const TransformablePtr<T>& ptr) = default;

        void promoteToStrongReference() { m_strongPtr = m_weakPtr; }
        void demoteToWeakReference() { m_strongPtr = nullptr; }
        bool isStrongReference() const { return m_strongPtr != nullptr; }

        T& operator*() const { return *m_weakPtr; }

        T* operator->() const { return m_weakPtr; }

        T* Ptr() const { return m_weakPtr; }
        T* get() const { return m_weakPtr; }

        operator T*() const { return m_weakPtr; }
        operator RefPtr<T>() const { return m_weakPtr; }


        TransformablePtr<T>& operator=(T* ptr)
        {
            m_weakPtr = ptr;
            m_strongPtr = ptr;
            return *this;
        }
        template<typename U>
        TransformablePtr<T>& operator=(const RefPtr<U>& ptr)
        {
            m_weakPtr = ptr.Ptr();
            m_strongPtr = ptr;
            return *this;
        }
        
        HashCode getHashCode() const
        {
            // Note: We need a `RefPtr<T>` to hash the same as a `T*`,
            // so that a `T*` can be used as a key in a dictionary with
            // `RefPtr<T>` keys, and vice versa.
            //
            return Slang::getHashCode(m_weakPtr);
        }

        bool operator==(const T* ptr) const { return m_weakPtr == ptr; }

        bool operator!=(const T* ptr) const { return m_weakPtr != ptr; }

        bool operator==(RefPtr<T> const& ptr) const { return m_weakPtr == ptr.Ptr(); }

        bool operator!=(RefPtr<T> const& ptr) const { return m_weakPtr != ptr.Ptr(); }

        bool operator==(TransformablePtr<T> const& ptr) const { return m_weakPtr == ptr.m_weakPtr; }

        bool operator!=(TransformablePtr<T> const& ptr) const { return m_weakPtr != ptr.m_weakPtr; }
    };
}
#endif

#ifndef _MSC_VER
#ifndef SLANG_CORE_SECURE_CRT_H
#define SLANG_CORE_SECURE_CRT_H
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <assert.h>

#include <wchar.h>

inline void memcpy_s(void *dest, [[maybe_unused]] size_t destSize, const void * src, size_t count)
{
    assert(destSize >= count);
    memcpy(dest, src, count);
}

#define _TRUNCATE ((size_t)-1)
#define _stricmp strcasecmp

inline void fopen_s(FILE**f, const char * fileName, const char * mode)
{
	*f = fopen(fileName, mode);
}

inline size_t fread_s(void * buffer, [[maybe_unused]] size_t bufferSize, size_t elementSize, size_t count, FILE * stream)
{
    assert(bufferSize >= elementSize * count);
    return fread(buffer, elementSize, count, stream);
}

inline size_t wcsnlen_s(const wchar_t * str, size_t /*numberofElements*/)
{
	return wcslen(str);
}

inline size_t strnlen_s(const char * str, size_t numberOfElements)
{
#if defined( __CYGWIN__ )
    const char* cur = str;
    if (str)
    {
        const char*const end = str + numberOfElements;
        while (*cur && cur < end) cur++;
    }
    return size_t(cur - str);
#else
	return strnlen(str, numberOfElements);
#endif
}

__attribute__((format(printf, 3, 4)))
inline int sprintf_s(char * buffer, size_t sizeOfBuffer, const char * format, ...)
{
	va_list argptr;
	va_start(argptr, format);
	int rs = vsnprintf(buffer, sizeOfBuffer, format, argptr);
	va_end(argptr);
	return rs;
}

// A patch was submitted to GCC wchar_t support in 2001, so I'm sure we can
// enable this any day now...
// __attribute__((format(wprintf, 3, 4)))
inline int swprintf_s(wchar_t * buffer, size_t sizeOfBuffer, const wchar_t * format, ...)
{
	va_list argptr;
	va_start(argptr, format);
	int rs = vswprintf(buffer, sizeOfBuffer, format, argptr);
	va_end(argptr);
	return rs;
}

inline void wcscpy_s(wchar_t * strDestination, size_t /*numberOfElements*/, const wchar_t * strSource)
{
	wcscpy(strDestination, strSource);
}
inline void strcpy_s(char * strDestination, size_t /*numberOfElements*/, const char * strSource)
{
	strcpy(strDestination, strSource);
}

inline void wcsncpy_s(wchar_t * strDestination, size_t /*numberOfElements*/, const wchar_t * strSource, size_t count)
{
	wcsncpy(strDestination, strSource, count);
}
inline void strncpy_s(char * strDestination, size_t /*numberOfElements*/, const char * strSource, size_t count)
{
	strncpy(strDestination, strSource, count);
}
#endif
#endif

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace Slang
{
    //
    // Types
    //

    struct StableHashCode64
    {
        uint64_t hash;
        explicit operator uint64_t() const { return hash; }
        bool operator==(StableHashCode64 other) const { return other.hash == hash; };
        bool operator!=(StableHashCode64 other) const { return other.hash != hash; };
    };

    struct StableHashCode32
    {
        uint32_t hash;
        explicit operator uint32_t() const { return hash; }
        bool operator==(StableHashCode32 other) const { return other.hash == hash; };
        bool operator!=(StableHashCode32 other) const { return other.hash != hash; };
    };

    /* The 'Stable' hash code functions produce hashes that must be

    * The same result for the same inputs on all targets
    * Rarely change - as their values can change the output of the Slang API/Serialization

    Hash value used from the 'Stable' functions can also be used as part of serialization -
    so it is in effect part of the API.

    In effect this means changing a 'Stable' algorithm will typically require doing a new release.
    */
    inline StableHashCode64 getStableHashCode64(const char* buffer, size_t numChars)
    {
        uint64_t hash = 0;
        for (size_t i = 0; i < numChars; ++i)
        {
            hash = uint64_t(buffer[i]) + (hash << 6) + (hash << 16) - hash;
        }
        return StableHashCode64{hash};
    }

    template<typename T>
    inline StableHashCode64 getStableHashCode64(const T& t)
    {
        static_assert(std::has_unique_object_representations_v<T>);
        return getStableHashCode64(reinterpret_cast<const char*>(&t), sizeof(T));
    }

    inline StableHashCode32 getStableHashCode32(const char* buffer, size_t numChars)
    {
        uint32_t hash = 0;
        for (size_t i = 0; i < numChars; ++i)
        {
            hash = uint32_t(buffer[i]) + (hash << 6) + (hash << 16) - hash;
        }
        return StableHashCode32{hash};
    }

    template<typename T>
    inline StableHashCode32 getStableHashCode32(const T& t)
    {
        static_assert(std::has_unique_object_representations_v<T>);
        return getStableHashCode32(reinterpret_cast<const char*>(&t), sizeof(T));
    }

    inline StableHashCode64 combineStableHash(StableHashCode64 h)
    {
        return h;
    }

    inline StableHashCode32 combineStableHash(StableHashCode32 h)
    {
        return h;
    }

    // A left fold with a mixing operation
    template<typename H, typename... Hs>
    H combineStableHash(H n, H m, Hs... args)
    {
        return combineStableHash(H{(n.hash * 16777619) ^ m.hash}, args...);
    }
}

// > Please draw a small horse in ASCII art:
//
//           ,~~.
//          (  9 )-_,
//  (\___ )=='-' )
//   \ .   ) )  /
//    \ `-' /  /
// ~'`~'`~'`~'`~
//


#include <new>
#include <type_traits>

namespace Slang
{
    class _EndLine
    {};
    extern _EndLine EndLine;

    // in-place reversion, works only for ascii string
    inline void reverseInplaceAscii(char* buffer, int length)
    {
        int i, j;
        char c;
        for (i = 0, j = length - 1; i<j; i++, j--)
        {
            c = buffer[i];
            buffer[i] = buffer[j];
            buffer[j] = c;
        }
    }
    template<typename IntType>
    inline int intToAscii(char* buffer, IntType val, int radix, int padTo = 0)
    {
        static_assert(std::is_integral_v<IntType>);

        int i = 0;
        IntType sign;
        
        sign = val;
        if (sign < 0)
        {
            val = (IntType)(0 - val);
        }

        do
        {
            int digit = (val % radix);
            if (digit <= 9)
                buffer[i++] = (char)(digit + '0');
            else
                buffer[i++] = (char)(digit - 10 + 'A');
        } while ((val /= radix) > 0);

        SLANG_ASSERT(i >= 0);
        while(i < padTo)
            buffer[i++] = '0';

        if (sign < 0)
            buffer[i++] = '-';

        // Put in normal character order
        reverseInplaceAscii(buffer, i);

        buffer[i] = '\0';
        return i;
    }

    SLANG_FORCE_INLINE bool isUtf8LeadingByte(char ch)
    {
        return (((unsigned char)ch) & 0xC0) == 0xC0;
    }

    SLANG_FORCE_INLINE bool isUtf8ContinuationByte(char ch)
    {
        return (((unsigned char)ch) & 0xC0) == 0x80;
    }

    /* A string slice that doesn't own the contained characters.
    It is the responsibility of code using the type to keep the memory backing 
    the slice in scope.
    A slice is generally *not* zero terminated. */
    struct SLANG_RT_API UnownedStringSlice
    {
    public:
        typedef UnownedStringSlice ThisType;

            // Type to indicate that a ctor is with a length to disabmiguate 0/nullptr 
            // causing ambiguity.
        struct WithLength {};

        UnownedStringSlice()
            : m_begin(nullptr)
            , m_end(nullptr)
        {}

        explicit UnownedStringSlice(char const* a) :
            m_begin(a),
            m_end(a ? a + strlen(a) : nullptr)
        {}
        UnownedStringSlice(char const* b, char const* e)
            : m_begin(b)
            , m_end(e)
        {}
        UnownedStringSlice(char const* b, size_t len)
            : m_begin(b)
            , m_end(b + len)
        {}
        UnownedStringSlice(WithLength, char const* b, size_t len)
            : m_begin(b)
            , m_end(b + len)
        {}

        SLANG_FORCE_INLINE char const* begin() const { return m_begin; }

        SLANG_FORCE_INLINE char const* end() const { return m_end; }

            /// True if slice is strictly contained in memory.
        bool isMemoryContained(const UnownedStringSlice& slice) const
        {
            return slice.m_begin >= m_begin && slice.m_end <= m_end; 
        }
        bool isMemoryContained(const char* pos) const
        {
            return pos >= m_begin && pos <= m_end;
        }

            /// Get the length in *bytes*
        Count getLength() const { return Index(m_end - m_begin); }

            /// Finds first index of char 'c'. If not found returns -1.
        Index indexOf(char c) const;
            /// Find first index of slice. If not found returns -1
        Index indexOf(const UnownedStringSlice& slice) const;

            /// Returns a substring. idx is the start index, and len
            /// is the amount of characters.
            /// The returned length might be truncated, if len extends beyond slice.
        UnownedStringSlice subString(Index idx, Index len) const;

            /// Return a head of the slice - everything up to the index
        SLANG_FORCE_INLINE UnownedStringSlice head(Index idx) const { SLANG_ASSERT(idx >= 0 && idx <= getLength()); return UnownedStringSlice(m_begin, idx); }
            /// Return a tail of the slice - everything from the index to the end of the slice
        SLANG_FORCE_INLINE UnownedStringSlice tail(Index idx) const { SLANG_ASSERT(idx >= 0 && idx <= getLength()); return UnownedStringSlice(m_begin + idx, m_end); }

            /// True if rhs and this are equal without having to take into account case
            /// Note 'case' here is *not* locale specific - it is only A-Z and a-z
        bool caseInsensitiveEquals(const ThisType& rhs) const;

        Index lastIndexOf(char c) const
        {
            const Index size = Index(m_end - m_begin);
            for (Index i = size - 1; i >= 0; --i)
            {
                if (m_begin[i] == c)
                {
                    return i;
                }
            }
            return -1;
        }

        const char& operator[](Index i) const
        {
            assert(i >= 0 && i < Index(m_end - m_begin));
            return m_begin[i];
        }

        bool operator==(ThisType const& other) const;
        bool operator!=(UnownedStringSlice const& other) const { return !(*this == other);  }

        bool operator==(char const* str) const { return (*this) == UnownedStringSlice(str); }
        bool operator!=(char const* str) const { return !(*this == str); }

            /// True if contents is a single char of c
        SLANG_FORCE_INLINE bool isChar(char c) const { return getLength() == 1 && m_begin[0] == c; }

        bool startsWithCaseInsensitive(UnownedStringSlice const& other) const;
        bool startsWith(UnownedStringSlice const& other) const;
        bool startsWith(char const* str) const;

        bool endsWithCaseInsensitive(UnownedStringSlice const& other) const;
        bool endsWithCaseInsensitive(char const* str) const;

        bool endsWith(UnownedStringSlice const& other) const;
        bool endsWith(char const* str) const;

            /// Trims any horizontal whitespace from the start and end and returns as a substring 
        UnownedStringSlice trim() const;
            /// Trims any 'c' from the start or the end, and returns as a substring
        UnownedStringSlice trim(char c) const;

            /// Trims any horizonatl whitespace from start and returns as a substring
        UnownedStringSlice trimStart() const;

        static constexpr bool kHasUniformHash = true;
        HashCode64 getHashCode() const
        {
            return Slang::getHashCode(m_begin, size_t(m_end - m_begin)); 
        }

        template <size_t SIZE> 
        SLANG_FORCE_INLINE static UnownedStringSlice fromLiteral(const char (&in)[SIZE]) { return UnownedStringSlice(in, SIZE - 1); }

    protected:

        char const* m_begin;
        char const* m_end;
    };

    // A more convenient way to make slices from *string literals*
    template <size_t SIZE>
    SLANG_FORCE_INLINE UnownedStringSlice toSlice(const char (&in)[SIZE]) { return UnownedStringSlice(in, SIZE - 1); }

    /// Same as UnownedStringSlice, but must be zero terminated. 
    /// Zero termination is *not* included in the length.
    struct SLANG_RT_API UnownedTerminatedStringSlice : public UnownedStringSlice
    {
    public:
        typedef UnownedStringSlice Super;
        typedef UnownedTerminatedStringSlice ThisType;

            /// We can turn into a regular zero terminated string
        SLANG_FORCE_INLINE operator const char*() const { return m_begin; }

            /// Exists to match the equivalent function in String.
        SLANG_FORCE_INLINE char const* getBuffer() const { return m_begin; }

            /// Construct from a literal directly.
        template <size_t SIZE>
        SLANG_FORCE_INLINE static ThisType fromLiteral(const char(&in)[SIZE]) { return ThisType(in, SIZE - 1); }

            /// Default constructor
        UnownedTerminatedStringSlice():Super(Super::WithLength(), "", 0) {}
        
            /// Note, b cannot be null because if it were then the string would not be null terminated
        UnownedTerminatedStringSlice(char const* b)
            : Super(b, b + strlen(b))
        {}
        UnownedTerminatedStringSlice(char const* b, size_t len)
            : Super(b, len)
        {
            // b must be valid and it must be null terminated
            SLANG_ASSERT(b && b[len] == 0);
        }
    };

    // A more convenient way to make terminated slices from *string literals*
    template <size_t SIZE>
    SLANG_FORCE_INLINE UnownedTerminatedStringSlice toTerminatedSlice(const char(&in)[SIZE]) { return UnownedTerminatedStringSlice(in, SIZE - 1); }

    // A `StringRepresentation` provides the backing storage for
    // all reference-counted string-related types.
    class SLANG_RT_API StringRepresentation : public RefObject
    {
    public:
        Index length;
        Index capacity;

        SLANG_FORCE_INLINE Index getLength() const
        {
            return length;
        }

        SLANG_FORCE_INLINE char* getData()
        {
            return (char*) (this + 1);
        }
        SLANG_FORCE_INLINE const char* getData() const
        {
            return (const char*)(this + 1);
        }

            /// Set the contents to be the slice. Must be enough capacity to hold the slice. 
        void setContents(const UnownedStringSlice& slice);

        static const char* getData(const StringRepresentation* stringRep)
        {
            return stringRep ? stringRep->getData() : "";
        }

        static UnownedStringSlice asSlice(const StringRepresentation* rep)
        {
            return rep ? UnownedStringSlice(rep->getData(), rep->getLength()) : UnownedStringSlice();
        }

        static bool equal(const StringRepresentation* a, const StringRepresentation* b)
        {
            return (a == b) || asSlice(a) == asSlice(b);
        }

        static StringRepresentation* createWithCapacityAndLength(Index capacity, Index length)
        {
            SLANG_ASSERT(capacity >= length);
            void* allocation = operator new(sizeof(StringRepresentation) + capacity + 1);
            StringRepresentation* obj = new(allocation) StringRepresentation();
            obj->capacity = capacity;
            obj->length = length;
            obj->getData()[length] = 0;
            return obj;
        }

        static StringRepresentation* createWithCapacity(Index capacity)
        {
            return createWithCapacityAndLength(capacity, 0);
        }

        static StringRepresentation* createWithLength(Index length)
        {
            return createWithCapacityAndLength(length, length);
        }

            /// Create a representation from the slice. If slice is empty will return nullptr.
        static StringRepresentation* create(const UnownedStringSlice& slice);
            /// Same as create, but representation will have refcount of 1 (if not nullptr)
        static StringRepresentation* createWithReference(const UnownedStringSlice& slice);

        StringRepresentation* cloneWithCapacity(Index newCapacity)
        {
            StringRepresentation* newObj = createWithCapacityAndLength(newCapacity, length);
            memcpy(getData(), newObj->getData(), length + 1);
            return newObj;
        }

        StringRepresentation* clone()
        {
            return cloneWithCapacity(length);
        }

        StringRepresentation* ensureCapacity(Index required)
        {
            if (capacity >= required) return this;

            Index newCapacity = capacity;
            if (!newCapacity) newCapacity = 16; // TODO: figure out good value for minimum capacity

            while (newCapacity < required)
            {
                newCapacity = 2 * newCapacity;
            }

            return cloneWithCapacity(newCapacity);
        }

            /// Overload delete to silence ASAN new-delete-type-mismatch errors.
            /// These occur because the allocation size of StringRepresentation
            /// does not match deallocation size (due variable sized string payload).
        void operator delete(void* p)
        {
            StringRepresentation* str = (StringRepresentation*) p;
            ::operator delete(str);
        }        
    };

    class String;

    struct SLANG_RT_API StringSlice
    {
    public:
        StringSlice();

        StringSlice(String const& str);

        StringSlice(String const& str, UInt beginIndex, UInt endIndex);

        UInt getLength() const
        {
            return endIndex - beginIndex;
        }

        char const* begin() const
        {
            return representation ? representation->getData() + beginIndex : "";
        }

        char const* end() const
        {
            return begin() + getLength();
        }

    private:
        RefPtr<StringRepresentation> representation;
        UInt beginIndex;
        UInt endIndex;

        friend class String;

        StringSlice(RefPtr<StringRepresentation> const& representation, UInt beginIndex, UInt endIndex)
            : representation(representation)
            , beginIndex(beginIndex)
            , endIndex(endIndex)
        {}
    };

    /// String as expected by underlying platform APIs
    class SLANG_RT_API OSString
    {
    public:
            /// Default
        OSString();
            /// NOTE! This assumes that begin is a new wchar_t[] buffer, and it will
            /// now be owned by the OSString
        OSString(wchar_t* begin, wchar_t* end);
            /// Move Ctor
        OSString(OSString&& rhs):
            m_begin(rhs.m_begin),
            m_end(rhs.m_end)
        {
            rhs.m_begin = nullptr;
            rhs.m_end = nullptr;
        }
            // Copy Ctor
        OSString(const OSString& rhs) :
            m_begin(nullptr),
            m_end(nullptr)
        {
            set(rhs.m_begin, rhs.m_end);
        }

            /// =
        void operator=(const OSString& rhs) { set(rhs.m_begin, rhs.m_end); }
        void operator=(OSString&& rhs)
        {
            auto begin = m_begin;
            auto end = m_end;
            m_begin = rhs.m_begin;
            m_end = rhs.m_end;
            rhs.m_begin = begin;
            rhs.m_end = end;
        }

        ~OSString() { _releaseBuffer(); }

        size_t getLength() const { return (m_end - m_begin); }
        void set(const wchar_t* begin, const wchar_t* end);

        operator wchar_t const*() const
        {
            return begin();
        }

        wchar_t const* begin() const;
        wchar_t const* end() const;

    private:

        void _releaseBuffer();

        wchar_t* m_begin;           ///< First character. This is a new wchar_t[] buffer
        wchar_t* m_end;             ///< Points to terminating 0
    };

    /*!
    @brief Represents a UTF-8 encoded string.
    */

    class SLANG_RT_API String
    {
        friend struct StringSlice;
        friend class StringBuilder;
    private:


        char* getData() const
        {
            return m_buffer ? m_buffer->getData() : (char*)"";
        }

     
        void ensureUniqueStorageWithCapacity(Index capacity);
     
        RefPtr<StringRepresentation> m_buffer;

    public:

        explicit String(StringRepresentation* buffer)
            : m_buffer(buffer)
        {}

        static String fromWString(const wchar_t* wstr);
        static String fromWString(const wchar_t* wstr, const wchar_t* wend);
        static String fromWChar(const wchar_t ch);
        static String fromUnicodePoint(Char32 codePoint);

        String()
        {
        }

            /// Returns a buffer which can hold at least count chars
        char* prepareForAppend(Index count);
            /// Append data written to buffer output via 'prepareForAppend' directly written 'inplace'
        void appendInPlace(const char* chars, Index count);

            /// Get the internal string represenation
        SLANG_FORCE_INLINE StringRepresentation* getStringRepresentation() const { return m_buffer; }

            /// Detach the representation (will leave string as empty). Rep ref count will remain unchanged.
        SLANG_FORCE_INLINE StringRepresentation* detachStringRepresentation() { return m_buffer.detach(); }

        const char* begin() const
        {
            return getData();
        }
        const char* end() const
        {
            return getData() + getLength();
        }

        void append(int32_t value, int radix = 10);
        void append(uint32_t value, int radix = 10);
        void append(int64_t value, int radix = 10);
        void append(uint64_t value, int radix = 10);
        void append(float val, const char* format = "%g");
        void append(double val, const char* format = "%g");

        // Padded hex representations
        void append(StableHashCode32 val);
        void append(StableHashCode64 val);

        void append(char const* str);
        void append(char const* str, size_t len);
        void append(const char* textBegin, char const* textEnd);
        void append(char chr);
        void append(String const& str);
        void append(StringSlice const& slice);
        void append(UnownedStringSlice const& slice);

            /// Append a character (to remove ambiguity with other integral types)
        void appendChar(char chr);

            /// Append the specified char count times
        void appendRepeatedChar(char chr, Index count);

        String(const char* str)
        {
            append(str);

        }
        String(const char* textBegin, char const* textEnd)
        {
            append(textBegin, textEnd);
        }

        // Make all String ctors from a numeric explicit, to avoid unexpected/unnecessary conversions
        explicit String(int32_t val, int radix = 10)
        {
            append(val, radix);
        }
        explicit String(uint32_t val, int radix = 10)
        {
            append(val, radix);
        }
        explicit String(int64_t val, int radix = 10)
        {
            append(val, radix);
        }
        explicit String(uint64_t val, int radix = 10)
        {
            append(val, radix);
        }
        explicit String(StableHashCode32 val)
        {
            append(val);
        }
        explicit String(StableHashCode64 val)
        {
            append(val);
        }
        explicit String(float val, const char* format = "%g")
        {
            append(val, format);
        }
        explicit String(double val, const char* format = "%g")
        {
            append(val, format);
        }

        explicit String(char chr)
        {
            appendChar(chr);
        }
        String(String const& str)
        {
            m_buffer = str.m_buffer;
        }
        String(String&& other)
        {
            m_buffer = _Move(other.m_buffer);
        }

        String(StringSlice const& slice)
        {
            append(slice);
        }

        String(UnownedStringSlice const& slice)
        {
            append(slice);
        }

        ~String()
        {
            m_buffer.setNull(); 
        }

        String& operator=(const String& str)
        {
            m_buffer = str.m_buffer;
            return *this;
        }
        String& operator=(String&& other)
        {
            m_buffer = _Move(other.m_buffer);
            return *this;
        }
        char operator[](Index id) const
        {
            SLANG_ASSERT(id >= 0 && id < getLength());
            // Silence a pedantic warning on GCC
#if __GNUC__
            if(id < 0) __builtin_unreachable();
#endif
            return begin()[id];
        }

        Index getLength() const
        {
            return m_buffer ? m_buffer->getLength() : 0;
        }
            /// Make the length of the string the amount specified. Must be less than current size
        void reduceLength(Index length);
        
        friend String operator+(const char*op1, const String & op2);
        friend String operator+(const String & op1, const char * op2);
        friend String operator+(const String & op1, const String & op2);

        StringSlice trimStart() const
        {
            if (!m_buffer)
                return StringSlice();
            Index startIndex = 0;
            const char*const data = getData();
            while (startIndex < getLength() &&
                (data[startIndex] == ' ' || data[startIndex] == '\t' || data[startIndex] == '\r' || data[startIndex] == '\n'))
                startIndex++;
            return StringSlice(m_buffer, startIndex, getLength());
        }

        StringSlice trimEnd() const
        {
            if (!m_buffer)
                return StringSlice();

            Index endIndex = getLength();
            const char*const data = getData();
            while (endIndex > 0 &&
                (data[endIndex-1] == ' ' || data[endIndex-1] == '\t' || data[endIndex-1] == '\r' || data[endIndex-1] == '\n'))
                endIndex--;

            return StringSlice(m_buffer, 0, endIndex);
        }

        StringSlice trim() const
        {
            if (!m_buffer)
                return StringSlice();

            Index startIndex = 0;
            const char*const data = getData();
            while (startIndex < getLength() &&
                (data[startIndex] == ' ' || data[startIndex] == '\t' || data[startIndex] == '\r' || data[startIndex] == '\n'))
                startIndex++;
            Index endIndex = getLength();
            while (endIndex > startIndex &&
                (data[endIndex-1] == ' ' || data[endIndex-1] == '\t' || data[endIndex-1] == '\r' || data[endIndex-1] == '\n'))
                endIndex--;

            return StringSlice(m_buffer, startIndex, endIndex);
        }

        StringSlice subString(Index id, Index len) const
        {
            if (len == 0)
                return StringSlice();

            if (id + len > getLength())
                len = getLength() - id;
#if _DEBUG
            if (id < 0 || id >= getLength() || (id + len) > getLength())
                SLANG_ASSERT_FAILURE("SubString: index out of range.");
            if (len < 0)
                SLANG_ASSERT_FAILURE("SubString: length less than zero.");
#endif
            return StringSlice(m_buffer, id, id + len);
        }

        char const* getBuffer() const
        {
            return getData();
        }

        OSString toWString(Index* len = 0) const;

        bool equals(const String& str, bool caseSensitive = true)
        {
            if (caseSensitive)
                return (strcmp(begin(), str.begin()) == 0);
            else
            {
#ifdef _MSC_VER
                return (_stricmp(begin(), str.begin()) == 0);
#else
                return (strcasecmp(begin(), str.begin()) == 0);
#endif
            }
        }
        bool operator==(const char* strbuffer) const
        {
            return (strcmp(begin(), strbuffer) == 0);
        }

        bool operator==(const String& str) const
        {
            return (strcmp(begin(), str.begin()) == 0);
        }
        bool operator!=(const char* strbuffer) const
        {
            return (strcmp(begin(), strbuffer) != 0);
        }
        bool operator!=(const String& str) const
        {
            return (strcmp(begin(), str.begin()) != 0);
        }
        bool operator>(const String& str) const
        {
            return (strcmp(begin(), str.begin()) > 0);
        }
        bool operator<(const String& str) const
        {
            return (strcmp(begin(), str.begin()) < 0);
        }
        bool operator>=(const String& str) const
        {
            return (strcmp(begin(), str.begin()) >= 0);
        }
        bool operator<=(const String& str) const
        {
            return (strcmp(begin(), str.begin()) <= 0);
        }

        SLANG_FORCE_INLINE bool operator==(const UnownedStringSlice& slice) const { return getUnownedSlice() == slice; }
        SLANG_FORCE_INLINE bool operator!=(const UnownedStringSlice& slice) const { return getUnownedSlice() != slice; }

        String toUpper() const
        {
            String result;
            for (auto c : *this)
            {
                char d = (c >= 'a' && c <= 'z') ? (c - ('a' - 'A')) : c;
                result.append(d);
            }
            return result;
        }

        String toLower() const
        {
            String result;
            for (auto c : *this)
            {
                char d = (c >= 'A' && c <= 'Z') ? (c - ('A' - 'a')) : c;
                result.append(d);
            }
            return result;
        }

        Index indexOf(const char* str, Index id) const // String str
        {
            if (id >= getLength())
                return Index(-1);
            auto findRs = strstr(begin() + id, str);
            Index res = findRs ? findRs - begin() : Index(-1);
            return res;
        }

        Index indexOf(const String& str, Index id) const
        {
            return indexOf(str.begin(), id);
        }

        Index indexOf(const char* str) const
        {
            return indexOf(str, 0);
        }

        Index indexOf(const String& str) const
        {
            return indexOf(str.begin(), 0);
        }

        void swapWith(String& other)
        {
            m_buffer.swapWith(other.m_buffer);
        }

        Index indexOf(char ch, Index id) const
        {
            const Index length = getLength();
            SLANG_ASSERT(id >= 0 && id <= length);

            if (!m_buffer)
                return Index(-1);

            const char* data = getData();
            for (Index i = id; i < length; i++)
                if (data[i] == ch)
                    return i;
            return Index(-1);
        }

        Index indexOf(char ch) const
        {
            return indexOf(ch, 0);
        }

        Index lastIndexOf(char ch) const
        {            
            const Index length = getLength();
            const char* data = getData();

            for (Index i = length - 1; i >= 0; --i)
                if (data[i] == ch)
                    return i;
            return Index(-1);
        }

        bool startsWith(const char* str) const 
        {
            if (!m_buffer)
                return false;
            Index strLen = Index(::strlen(str));
            if (strLen > getLength())
                return false;

            const char*const data = getData();

            for (Index i = 0; i < strLen; i++)
                if (str[i] != data[i])
                    return false;
            return true;
        }

        bool startsWith(const String& str) const
        {
            return startsWith(str.begin());
        }

        bool endsWith(char const* str)  const // String str
        {
            if (!m_buffer)
                return false;

            const Index strLen = Index(::strlen(str));
            const Index len = getLength();

            if (strLen > len)
                return false;
            const char* data = getData();
            for (Index i = strLen; i > 0; i--)
                if (str[i-1] != data[len - strLen + i-1])
                    return false;
            return true;
        }

        bool endsWith(const String& str) const
        {
            return endsWith(str.begin());
        }

        bool contains(const char* str) const // String str
        {
            return m_buffer && indexOf(str) != Index(-1); 
        }

        bool contains(const String& str) const
        {
            return contains(str.begin());
        }

        static constexpr bool kHasUniformHash = true;
        HashCode64 getHashCode() const
        {
            return Slang::getHashCode(StringRepresentation::asSlice(m_buffer));
        }

        UnownedStringSlice getUnownedSlice() const
        {
            return StringRepresentation::asSlice(m_buffer);
        }
    };

    class SLANG_RT_API StringBuilder : public String
    {
    private:
        enum { InitialSize = 1024 };
    public:
        typedef String Super;
        using Super::append;

        explicit StringBuilder(UInt bufferSize = InitialSize)
        {
            ensureUniqueStorageWithCapacity(bufferSize);
        }

        void ensureCapacity(UInt size)
        {
            ensureUniqueStorageWithCapacity(size);
        }
        StringBuilder& operator << (char ch)
        {
            appendChar(ch);
            return *this;
        }
        StringBuilder& operator << (Int32 val)
        {
            append(val);
            return *this;
        }
        StringBuilder& operator << (UInt32 val)
        {
            append(val);
            return *this;
        }
        StringBuilder& operator << (Int64 val)
        {
            append(val);
            return *this;
        }
        StringBuilder& operator << (UInt64 val)
        {
            append(val);
            return *this;
        }
        StringBuilder& operator << (float val)
        {
            append(val);
            return *this;
        }
        StringBuilder& operator << (double val)
        {
            append(val);
            return *this;
        }
        StringBuilder& operator << (const char* str)
        {
            append(str, strlen(str));
            return *this;
        }
        StringBuilder& operator << (const String& str)
        {
            append(str);
            return *this;
        }
        StringBuilder& operator << (UnownedStringSlice const& str)
        {
            append(str);
            return *this;
        }
        StringBuilder& operator << (const _EndLine)
        {
            appendChar('\n');
            return *this;
        }

        String toString()
        {
            return *this;
        }

        String produceString()
        {
            return *this;
        }

#if 0
        void Remove(int id, int len)
        {
#if _DEBUG
            if (id >= length || id < 0)
                SLANG_ASSERT_FAILURE("Remove: Index out of range.");
            if (len < 0)
                SLANG_ASSERT_FAILURE("Remove: remove length smaller than zero.");
#endif
            int actualDelLength = ((id + len) >= length) ? (length - id) : len;
            for (int i = id + actualDelLength; i <= length; i++)
                buffer[i - actualDelLength] = buffer[i];
            length -= actualDelLength;
        }
#endif
        friend std::ostream& operator<< (std::ostream& stream, const String& s);

        void clear()
        {
            m_buffer.setNull();
        }
    };

    int stringToInt(const String& str, int radix = 10);
    unsigned int stringToUInt(const String& str, int radix = 10);
    double stringToDouble(const String& str);
    float stringToFloat(const String& str);
}

std::ostream& operator<< (std::ostream& stream, const Slang::String& s);

#endif


#if defined(_MSC_VER)
#   define SLANG_PRELUDE_SHARED_LIB_EXPORT __declspec(dllexport)
#else
#   define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__((__visibility__("default")))
//#   define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__ ((dllexport)) __attribute__((__visibility__("default")))
#endif    

#ifdef __cplusplus    
#   define SLANG_PRELUDE_EXTERN_C extern "C"
#   define SLANG_PRELUDE_EXTERN_C_START extern "C" {
#   define SLANG_PRELUDE_EXTERN_C_END }
#else
#   define SLANG_PRELUDE_EXTERN_C 
#   define SLANG_PRELUDE_EXTERN_C_START
#   define SLANG_PRELUDE_EXTERN_C_END 
#endif    

#define SLANG_PRELUDE_NAMESPACE

#ifndef SLANG_NO_THROW
#   define SLANG_NO_THROW
#endif
#ifndef SLANG_STDCALL
#   define SLANG_STDCALL
#endif
#ifndef SLANG_MCALL
#   define SLANG_MCALL SLANG_STDCALL
#endif
#ifndef SLANG_FORCE_INLINE
#    define SLANG_FORCE_INLINE inline
#endif
#ifndef SLANG_PRELUDE_CPP_TYPES_CORE_H
#define SLANG_PRELUDE_CPP_TYPES_CORE_H

#ifndef SLANG_PRELUDE_ASSERT
#   ifdef SLANG_PRELUDE_ENABLE_ASSERT
#       define SLANG_PRELUDE_ASSERT(VALUE) assert(VALUE)
#   else
#       define SLANG_PRELUDE_ASSERT(VALUE) 
#   endif
#endif

// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count)  SLANG_PRELUDE_ASSERT(index < count); 
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0; 
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) index = (index <= (sizeInBytes - elemSize)) ? index : 0; 

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If SLANG_ENABLE_BOUND_ZERO_INDEX
// the fix macro will zero the index, if out of range
#ifdef  SLANG_ENABLE_BOUND_ZERO_INDEX
#   define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#   define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#   define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#   define SLANG_BOUND_FIX(index, count) 
#   define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) 
#   define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) 
#endif

#ifndef SLANG_BOUND_CHECK
#   define SLANG_BOUND_CHECK(index, count) SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#   define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#   define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

struct TypeInfo
{
    size_t typeSize;
};

template <typename T, size_t SIZE>
struct FixedArray
{
    const T& operator[](size_t index) const { SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE); return m_data[index]; }
    T& operator[](size_t index) { SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE); return m_data[index]; }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can potentially 
// do bounds checking.  
template <typename T>
struct Array
{
    const T& operator[](size_t index) const { SLANG_BOUND_CHECK(index, count); return data[index]; }
    T& operator[](size_t index) { SLANG_BOUND_CHECK(index, count); return data[index]; }

    T* data;
    size_t count;
};

/* Constant buffers become a pointer to the contained type, so ConstantBuffer<T> becomes T* in C++ code.
*/

template <typename T, int COUNT>
struct Vector;

template <typename T>
struct Vector<T, 1>
{
    T x;
    const T& operator[](size_t /*index*/) const { return x; }
    T& operator[](size_t /*index*/) { return x; }
    operator T() const { return x; }
    Vector() = default;
    Vector(T scalar)
    {
        x = scalar;
    }
    template <typename U>
    Vector(Vector<U, 1> other)
    {
        x = (T)other.x;
    }
    template <typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 1;
        if (otherSize < minSize) minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template <typename T>
struct Vector<T, 2>
{
    T x, y;
    const T& operator[](size_t index) const { return index == 0 ? x : y; }
    T& operator[](size_t index) { return index == 0 ? x : y; }
    Vector() = default;
    Vector(T scalar)
    {
        x = y = scalar;
    }
    Vector(T _x, T _y)
    {
        x = _x;
        y = _y;
    }
    template <typename U>
    Vector(Vector<U, 2> other)
    {
        x = (T)other.x;
        y = (T)other.y;
    }
    template <typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 2;
        if (otherSize < minSize) minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template <typename T>
struct Vector<T, 3>
{
    T x, y, z;
    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }

    Vector() = default;
    Vector(T scalar)
    {
        x = y = z = scalar;
    }
    Vector(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    template <typename U>
    Vector(Vector<U, 3> other)
    {
        x = (T)other.x;
        y = (T)other.y;
        z = (T)other.z;
    }
    template <typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 3;
        if (otherSize < minSize) minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template <typename T>
struct Vector<T, 4>
{
    T x, y, z, w;

    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }
    Vector() = default;
    Vector(T scalar)
    {
        x = y = z = w = scalar;
    }
    Vector(T _x, T _y, T _z, T _w)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    template <typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 4;
        if (otherSize < minSize) minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
 
};

template<typename T, int N>
SLANG_FORCE_INLINE Vector<T, N> _slang_select(Vector<bool, N> condition, Vector<T, N> v0, Vector<T, N> v1)
{
    Vector<T, N> result;
    for (int i = 0; i < N; i++)
    {
        result[i] = condition[i] ? v0[i] : v1[i];
    }
    return result;
}

template<typename T>
SLANG_FORCE_INLINE T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

template<typename T, int N>
SLANG_FORCE_INLINE T _slang_vector_get_element(Vector<T, N> x, int index)
{
    return x[index];
}

template<typename T, int N>
SLANG_FORCE_INLINE const T* _slang_vector_get_element_ptr(const Vector<T, N>* x, int index)
{
    return &((*const_cast<Vector<T,N>*>(x))[index]);
}

template<typename T, int N>
SLANG_FORCE_INLINE T* _slang_vector_get_element_ptr(Vector<T, N>* x, int index)
{
    return &((*x)[index]);
}

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

typedef uint32_t uint;

#define SLANG_VECTOR_BINARY_OP(T, op) \
    template<int n> \
    SLANG_FORCE_INLINE Vector<T, n> operator op(const Vector<T, n>& thisVal, const Vector<T, n>& other) \
    { \
        Vector<T, n> result;\
        for (int i = 0; i < n; i++) \
            result[i] = thisVal[i] op other[i]; \
        return result;\
    }
#define SLANG_VECTOR_BINARY_COMPARE_OP(T, op) \
    template<int n> \
    SLANG_FORCE_INLINE Vector<bool, n> operator op(const Vector<T, n>& thisVal, const Vector<T, n>& other) \
    { \
        Vector<bool, n> result;\
        for (int i = 0; i < n; i++) \
            result[i] = thisVal[i] op other[i]; \
        return result;\
    }

#define SLANG_VECTOR_UNARY_OP(T, op) \
    template<int n> \
    SLANG_FORCE_INLINE Vector<T, n> operator op(const Vector<T, n>& thisVal) \
    { \
        Vector<T, n> result;\
        for (int i = 0; i < n; i++) \
            result[i] = op thisVal[i]; \
        return result;\
    }
#define SLANG_INT_VECTOR_OPS(T) \
    SLANG_VECTOR_BINARY_OP(T, +)\
    SLANG_VECTOR_BINARY_OP(T, -)\
    SLANG_VECTOR_BINARY_OP(T, *)\
    SLANG_VECTOR_BINARY_OP(T, / )\
    SLANG_VECTOR_BINARY_OP(T, &)\
    SLANG_VECTOR_BINARY_OP(T, |)\
    SLANG_VECTOR_BINARY_OP(T, &&)\
    SLANG_VECTOR_BINARY_OP(T, ||)\
    SLANG_VECTOR_BINARY_OP(T, ^)\
    SLANG_VECTOR_BINARY_OP(T, %)\
    SLANG_VECTOR_BINARY_OP(T, >>)\
    SLANG_VECTOR_BINARY_OP(T, <<)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=)\
    SLANG_VECTOR_UNARY_OP(T, !)\
    SLANG_VECTOR_UNARY_OP(T, ~)
#define SLANG_FLOAT_VECTOR_OPS(T) \
    SLANG_VECTOR_BINARY_OP(T, +)\
    SLANG_VECTOR_BINARY_OP(T, -)\
    SLANG_VECTOR_BINARY_OP(T, *)\
    SLANG_VECTOR_BINARY_OP(T, /)\
    SLANG_VECTOR_UNARY_OP(T, -)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=)

SLANG_INT_VECTOR_OPS(bool)
SLANG_INT_VECTOR_OPS(int)
SLANG_INT_VECTOR_OPS(int8_t)
SLANG_INT_VECTOR_OPS(int16_t)
SLANG_INT_VECTOR_OPS(int64_t)
SLANG_INT_VECTOR_OPS(uint)
SLANG_INT_VECTOR_OPS(uint8_t)
SLANG_INT_VECTOR_OPS(uint16_t)
SLANG_INT_VECTOR_OPS(uint64_t)

SLANG_FLOAT_VECTOR_OPS(float)
SLANG_FLOAT_VECTOR_OPS(double)

#define SLANG_VECTOR_INT_NEG_OP(T) \
    template<int N>\
    Vector<T, N> operator-(const Vector<T, N>& thisVal) \
    { \
        Vector<T, N> result;\
        for (int i = 0; i < N; i++) \
            result[i] = 0 - thisVal[i]; \
        return result;\
    }
SLANG_VECTOR_INT_NEG_OP(int)
SLANG_VECTOR_INT_NEG_OP(int8_t)
SLANG_VECTOR_INT_NEG_OP(int16_t)
SLANG_VECTOR_INT_NEG_OP(int64_t)
SLANG_VECTOR_INT_NEG_OP(uint)
SLANG_VECTOR_INT_NEG_OP(uint8_t)
SLANG_VECTOR_INT_NEG_OP(uint16_t)
SLANG_VECTOR_INT_NEG_OP(uint64_t)

#define SLANG_FLOAT_VECTOR_MOD(T)\
    template<int N> \
    Vector<T, N> operator%(const Vector<T, N>& left, const Vector<T, N>& right) \
    {\
        Vector<T, N> result;\
        for (int i = 0; i < N; i++) \
            result[i] = _slang_fmod(left[i], right[i]); \
        return result;\
    }

SLANG_FLOAT_VECTOR_MOD(float)
SLANG_FLOAT_VECTOR_MOD(double)
#undef SLANG_FLOAT_VECTOR_MOD
#undef SLANG_VECTOR_BINARY_OP
#undef SLANG_VECTOR_UNARY_OP
#undef SLANG_INT_VECTOR_OPS
#undef SLANG_FLOAT_VECTOR_OPS
#undef SLANG_VECTOR_INT_NEG_OP
#undef SLANG_FLOAT_VECTOR_MOD

template <typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    Vector<T, COLS>& operator[](size_t index) { return rows[index]; }
    Matrix() = default;
    Matrix(T scalar)
    {
        for (int i = 0; i < ROWS; i++)
            rows[i] = Vector<T, COLS>(scalar);
    }
    Matrix(const Vector<T, COLS>& row0)
    {
        rows[0] = row0;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1)
    {
        rows[0] = row0;
        rows[1] = row1;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2, const Vector<T, COLS>& row3)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
        rows[3] = row3;
    }
    template<typename U, int otherRow, int otherCol>
    Matrix(const Matrix<U, otherRow, otherCol>& other)
    {
        int minRow = ROWS;
        int minCol = COLS;
        if (minRow > otherRow) minRow = otherRow;
        if (minCol > otherCol) minCol = otherCol;
        for (int i = 0; i < minRow; i++)
            for (int j = 0; j < minCol; j++)
                rows[i][j] = (T)other.rows[i][j];
    }
    Matrix(T v0, T v1, T v2, T v3)
    {
        rows[0][0] = v0;  rows[0][1] = v1;
        rows[1][0] = v2;  rows[1][1] = v3;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5)
    {
        if (COLS == 3)
        {
            rows[0][0] = v0;  rows[0][1] = v1; rows[0][2] = v2;
            rows[1][0] = v3;  rows[1][1] = v4; rows[1][2] = v5;
        }
        else
        {
            rows[0][0] = v0;  rows[0][1] = v1;
            rows[1][0] = v2;  rows[1][1] = v3;
            rows[2][0] = v4;  rows[2][1] = v5;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;  rows[0][1] = v1; rows[0][2] = v2; rows[0][3] = v3;
            rows[1][0] = v4;  rows[1][1] = v5; rows[1][2] = v6; rows[1][3] = v7;
        }
        else
        {
            rows[0][0] = v0;  rows[0][1] = v1;
            rows[1][0] = v2;  rows[1][1] = v3;
            rows[2][0] = v4;  rows[2][1] = v5;
            rows[3][0] = v6;  rows[3][1] = v7;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8)
    {
        rows[0][0] = v0;  rows[0][1] = v1;  rows[0][2] = v2;
        rows[1][0] = v3;  rows[1][1] = v4;  rows[1][2] = v5;
        rows[2][0] = v6;  rows[2][1] = v7;  rows[2][2] = v8;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;  rows[0][1] = v1;  rows[0][2] = v2;  rows[0][3] = v3;
            rows[1][0] = v4;  rows[1][1] = v5;  rows[1][2] = v6;  rows[1][3] = v7;
            rows[2][0] = v8;  rows[2][1] = v9;  rows[2][2] = v10; rows[2][3] = v11;
        }
        else
        {
            rows[0][0] = v0;  rows[0][1] = v1;  rows[0][2] = v2;
            rows[1][0] = v3;  rows[1][1] = v4;  rows[1][2] = v5;
            rows[2][0] = v6;  rows[2][1] = v7;  rows[2][2] = v8;
            rows[3][0] = v9;  rows[3][1] = v10; rows[3][2] = v11;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15)
    {
        rows[0][0] = v0;  rows[0][1] = v1;  rows[0][2] = v2;  rows[0][3] = v3;
        rows[1][0] = v4;  rows[1][1] = v5;  rows[1][2] = v6;  rows[1][3] = v7;
        rows[2][0] = v8;  rows[2][1] = v9;  rows[2][2] = v10; rows[2][3] = v11;
        rows[3][0] = v12; rows[3][1] = v13; rows[3][2] = v14; rows[3][3] = v15;
    }
};

#define SLANG_MATRIX_BINARY_OP(T, op) \
    template<int R, int C> \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j]; \
        return result;\
    }

#define SLANG_MATRIX_UNARY_OP(T, op) \
    template<int R, int C> \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                result[i].rows[i][j] = op thisVal.rows[i][j]; \
        return result;\
    }
#define SLANG_INT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)\
    SLANG_MATRIX_BINARY_OP(T, -)\
    SLANG_MATRIX_BINARY_OP(T, *)\
    SLANG_MATRIX_BINARY_OP(T, / )\
    SLANG_MATRIX_BINARY_OP(T, &)\
    SLANG_MATRIX_BINARY_OP(T, |)\
    SLANG_MATRIX_BINARY_OP(T, &&)\
    SLANG_MATRIX_BINARY_OP(T, ||)\
    SLANG_MATRIX_BINARY_OP(T, ^)\
    SLANG_MATRIX_BINARY_OP(T, %)\
    SLANG_MATRIX_UNARY_OP(T, !)\
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)\
    SLANG_MATRIX_BINARY_OP(T, -)\
    SLANG_MATRIX_BINARY_OP(T, *)\
    SLANG_MATRIX_BINARY_OP(T, /)\
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(int8_t)
SLANG_INT_MATRIX_OPS(int16_t)
SLANG_INT_MATRIX_OPS(int64_t)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(uint8_t)
SLANG_INT_MATRIX_OPS(uint16_t)
SLANG_INT_MATRIX_OPS(uint64_t)

SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)

#define SLANG_MATRIX_INT_NEG_OP(T) \
    template<int R, int C>\
    SLANG_FORCE_INLINE Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
            result.rows[i][j] = 0 - thisVal.rows[i][j]; \
        return result;\
    }
    SLANG_MATRIX_INT_NEG_OP(int)
    SLANG_MATRIX_INT_NEG_OP(int8_t)
    SLANG_MATRIX_INT_NEG_OP(int16_t)
    SLANG_MATRIX_INT_NEG_OP(int64_t)
    SLANG_MATRIX_INT_NEG_OP(uint)
    SLANG_MATRIX_INT_NEG_OP(uint8_t)
    SLANG_MATRIX_INT_NEG_OP(uint16_t)
    SLANG_MATRIX_INT_NEG_OP(uint64_t)

#define SLANG_FLOAT_MATRIX_MOD(T)\
    template<int R, int C> \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator%(Matrix<T, R, C> left, Matrix<T, R, C> right) \
    {\
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                result.rows[i][j] = _slang_fmod(left.rows[i][j], right.rows[i][j]); \
        return result;\
    }

    SLANG_FLOAT_MATRIX_MOD(float)
    SLANG_FLOAT_MATRIX_MOD(double)
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

template<typename TResult, typename TInput>
TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

#endif



#ifndef SLANG_PRELUDE_SCALAR_INTRINSICS_H
#define SLANG_PRELUDE_SCALAR_INTRINSICS_H

#if !defined(SLANG_LLVM) && SLANG_PROCESSOR_X86_64 && SLANG_VC
//  If we have visual studio and 64 bit processor, we can assume we have popcnt, and can include x86 intrinsics
#   include <intrin.h>
#endif

#ifndef SLANG_FORCE_INLINE
#    define SLANG_FORCE_INLINE inline
#endif

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE {
#endif

#ifndef SLANG_PRELUDE_PI
#   define SLANG_PRELUDE_PI           3.14159265358979323846
#endif


union Union32 
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

// 32 bit cast conversions
SLANG_FORCE_INLINE int32_t _bitCastFloatToInt(float f) { Union32 u; u.f = f; return u.i; }
SLANG_FORCE_INLINE float _bitCastIntToFloat(int32_t i) { Union32 u; u.i = i; return u.f; }
SLANG_FORCE_INLINE uint32_t _bitCastFloatToUInt(float f) { Union32 u; u.f = f; return u.u; }
SLANG_FORCE_INLINE float _bitCastUIntToFloat(uint32_t ui) { Union32 u; u.u = ui; return u.f; }

// ----------------------------- F16 -----------------------------------------


// This impl is based on FloatToHalf that is in Slang codebase
SLANG_FORCE_INLINE uint32_t f32tof16(const float value)
{
    const uint32_t inBits = _bitCastFloatToUInt(value);

    // bits initially set to just the sign bit
    uint32_t bits = (inBits >> 16) & 0x8000;
    // Mantissa can't be used as is, as it holds last bit, for rounding.
    uint32_t m = (inBits >> 12) & 0x07ff;
    uint32_t e = (inBits >> 23) & 0xff;

    if (e < 103)
    {
        // It's zero
        return bits;
    }
    if (e == 0xff)
    {
        // Could be a NAN or INF. Is INF if *input* mantissa is 0.
        
        // Remove last bit for rounding to make output mantissa.
        m >>= 1;
       
        // We *assume* float16/float32 signaling bit and remaining bits
        // semantics are the same. (The signalling bit convention is target specific!).
        // Non signal bit's usage within mantissa for a NAN are also target specific.
      
        // If the m is 0, it could be because the result is INF, but it could also be because all the 
        // bits that made NAN were dropped as we have less mantissa bits in f16. 
           
        // To fix for this we make non zero if m is 0 and the input mantissa was not.
        // This will (typically) produce a signalling NAN.
        m += uint32_t(m == 0 && (inBits & 0x007fffffu));
       
        // Combine for output
        return (bits | 0x7c00u | m);
    }
    if (e > 142)
    {
        // INF. 
        return bits | 0x7c00u;
    }
    if (e < 113)
    {
        m |= 0x0800u;
        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
        return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
}

static const float g_f16tof32Magic = _bitCastIntToFloat((127 + (127 - 15)) << 23);

SLANG_FORCE_INLINE float f16tof32(const uint32_t value)
{
    const uint32_t sign = (value & 0x8000) << 16;
    uint32_t exponent = (value & 0x7c00) >> 10;
    uint32_t mantissa = (value & 0x03ff);

    if (exponent == 0)
    {
        // If mantissa is 0 we are done, as output is 0. 
        // If it's not zero we must have a denormal.
        if (mantissa)
        {
            // We have a denormal so use the magic to do exponent adjust
            return _bitCastIntToFloat(sign | ((value & 0x7fff) << 13)) * g_f16tof32Magic;
        }
    }
    else 
    {
        // If the exponent is NAN or INF exponent is 0x1f on input. 
        // If that's the case, we just need to set the exponent to 0xff on output
        // and the mantissa can just stay the same. If its 0 it's INF, else it is NAN and we just copy the bits
        //
        // Else we need to correct the exponent in the normalized case.
        exponent = (exponent == 0x1F) ? 0xff : (exponent + (-15 + 127));
    }
    
    return _bitCastUIntToFloat(sign | (exponent << 23) | (mantissa << 13));
}

// ----------------------------- F32 -----------------------------------------

// Helpers
SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary 
float F32_ceil(float f);
float F32_floor(float f);
float F32_round(float f);
float F32_sin(float f);
float F32_cos(float f);
float F32_tan(float f);
float F32_asin(float f);
float F32_acos(float f);
float F32_atan(float f);
float F32_sinh(float f);
float F32_cosh(float f);
float F32_tanh(float f);
float F32_log2(float f);
float F32_log(float f);
float F32_log10(float f);
float F32_exp2(float f);
float F32_exp(float f);
float F32_abs(float f);
float F32_trunc(float f);
float F32_sqrt(float f);

bool F32_isnan(float f);
bool F32_isfinite(float f); 
bool F32_isinf(float f);

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b) { return a < b ? a : b; }
SLANG_FORCE_INLINE float F32_max(float a, float b) { return a > b ? a : b; }
float F32_pow(float a, float b);
float F32_fmod(float a, float b);
float F32_remainder(float a, float b);
float F32_atan2(float a, float b);

float F32_frexp(float x, int* e);

float F32_modf(float x, float* ip);

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c) { return a * b + c; }

SLANG_PRELUDE_EXTERN_C_END

#else

// Unary 
SLANG_FORCE_INLINE float F32_ceil(float f) { return ::ceilf(f); }
SLANG_FORCE_INLINE float F32_floor(float f) { return ::floorf(f); }
SLANG_FORCE_INLINE float F32_round(float f) { return ::roundf(f); }
SLANG_FORCE_INLINE float F32_sin(float f) { return ::sinf(f); }
SLANG_FORCE_INLINE float F32_cos(float f) { return ::cosf(f); }
SLANG_FORCE_INLINE float F32_tan(float f) { return ::tanf(f); }
SLANG_FORCE_INLINE float F32_asin(float f) { return ::asinf(f); }
SLANG_FORCE_INLINE float F32_acos(float f) { return ::acosf(f); }
SLANG_FORCE_INLINE float F32_atan(float f) { return ::atanf(f); }
SLANG_FORCE_INLINE float F32_sinh(float f) { return ::sinhf(f); }
SLANG_FORCE_INLINE float F32_cosh(float f) { return ::coshf(f); }
SLANG_FORCE_INLINE float F32_tanh(float f) { return ::tanhf(f); }
SLANG_FORCE_INLINE float F32_log2(float f) { return ::log2f(f); }
SLANG_FORCE_INLINE float F32_log(float f) { return ::logf(f); }
SLANG_FORCE_INLINE float F32_log10(float f) { return ::log10f(f); }
SLANG_FORCE_INLINE float F32_exp2(float f) { return ::exp2f(f); }
SLANG_FORCE_INLINE float F32_exp(float f) { return ::expf(f); }
SLANG_FORCE_INLINE float F32_abs(float f) { return ::fabsf(f); }
SLANG_FORCE_INLINE float F32_trunc(float f) { return ::truncf(f); }
SLANG_FORCE_INLINE float F32_sqrt(float f) { return ::sqrtf(f); }

SLANG_FORCE_INLINE bool F32_isnan(float f) { return SLANG_PRELUDE_STD isnan(f); }
SLANG_FORCE_INLINE bool F32_isfinite(float f) { return SLANG_PRELUDE_STD isfinite(f); }
SLANG_FORCE_INLINE bool F32_isinf(float f) { return SLANG_PRELUDE_STD isinf(f); }

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b) { return ::fminf(a, b); }
SLANG_FORCE_INLINE float F32_max(float a, float b) { return ::fmaxf(a, b); }
SLANG_FORCE_INLINE float F32_pow(float a, float b) { return ::powf(a, b); }
SLANG_FORCE_INLINE float F32_fmod(float a, float b) { return ::fmodf(a, b); }
SLANG_FORCE_INLINE float F32_remainder(float a, float b) { return ::remainderf(a, b); }
SLANG_FORCE_INLINE float F32_atan2(float a, float b) { return float(::atan2(a, b)); }

SLANG_FORCE_INLINE float F32_frexp(float x, int* e) { return ::frexpf(x, e); }

SLANG_FORCE_INLINE float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c) { return ::fmaf(a, b, c); }

#endif

SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1 
	float a = radians * (1.0f /  float(SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F32_floor(a);
    // Convert back to 0 - 2pi range
	return (a * float(SLANG_PRELUDE_PI * 2));
}

SLANG_FORCE_INLINE float F32_rsqrt(float f) { return 1.0f / F32_sqrt(f); }
SLANG_FORCE_INLINE float F32_sign(float f) { return ( f == 0.0f) ? f : (( f < 0.0f) ? -1.0f : 1.0f); } 
SLANG_FORCE_INLINE float F32_frac(float f) { return f - F32_floor(f); }

SLANG_FORCE_INLINE uint32_t F32_asuint(float f) { Union32 u; u.f = f; return u.u; }
SLANG_FORCE_INLINE int32_t F32_asint(float f) { Union32 u; u.f = f; return u.i; }

// ----------------------------- F64 -----------------------------------------

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary 
double F64_ceil(double f);
double F64_floor(double f);
double F64_round(double f);
double F64_sin(double f);
double F64_cos(double f);
double F64_tan(double f);
double F64_asin(double f);
double F64_acos(double f);
double F64_atan(double f);
double F64_sinh(double f);
double F64_cosh(double f);
double F64_tanh(double f);
double F64_log2(double f);
double F64_log(double f);
double F64_log10(double f);
double F64_exp2(double f);
double F64_exp(double f);
double F64_abs(double f);
double F64_trunc(double f);
double F64_sqrt(double f);

bool F64_isnan(double f);
bool F64_isfinite(double f);
bool F64_isinf(double f);

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b) { return a < b ? a : b; }
SLANG_FORCE_INLINE double F64_max(double a, double b) { return a > b ? a : b; }
double F64_pow(double a, double b);
double F64_fmod(double a, double b);
double F64_remainder(double a, double b);
double F64_atan2(double a, double b);

double F64_frexp(double x, int* e);

double F64_modf(double x, double* ip);

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c) { return a * b + c; }

SLANG_PRELUDE_EXTERN_C_END

#else // SLANG_LLVM

// Unary 
SLANG_FORCE_INLINE double F64_ceil(double f) { return ::ceil(f); }
SLANG_FORCE_INLINE double F64_floor(double f) { return ::floor(f); }
SLANG_FORCE_INLINE double F64_round(double f) { return ::round(f); }
SLANG_FORCE_INLINE double F64_sin(double f) { return ::sin(f); }
SLANG_FORCE_INLINE double F64_cos(double f) { return ::cos(f); }
SLANG_FORCE_INLINE double F64_tan(double f) { return ::tan(f); }
SLANG_FORCE_INLINE double F64_asin(double f) { return ::asin(f); }
SLANG_FORCE_INLINE double F64_acos(double f) { return ::acos(f); }
SLANG_FORCE_INLINE double F64_atan(double f) { return ::atan(f); }
SLANG_FORCE_INLINE double F64_sinh(double f) { return ::sinh(f); }
SLANG_FORCE_INLINE double F64_cosh(double f) { return ::cosh(f); }
SLANG_FORCE_INLINE double F64_tanh(double f) { return ::tanh(f); }
SLANG_FORCE_INLINE double F64_log2(double f) { return ::log2(f); }
SLANG_FORCE_INLINE double F64_log(double f) { return ::log(f); }
SLANG_FORCE_INLINE double F64_log10(float f) { return ::log10(f); }
SLANG_FORCE_INLINE double F64_exp2(double f) { return ::exp2(f); }
SLANG_FORCE_INLINE double F64_exp(double f) { return ::exp(f); }
SLANG_FORCE_INLINE double F64_abs(double f) { return ::fabs(f); }
SLANG_FORCE_INLINE double F64_trunc(double f) { return ::trunc(f); }
SLANG_FORCE_INLINE double F64_sqrt(double f) { return ::sqrt(f); }


SLANG_FORCE_INLINE bool F64_isnan(double f) { return SLANG_PRELUDE_STD isnan(f); }
SLANG_FORCE_INLINE bool F64_isfinite(double f) { return SLANG_PRELUDE_STD isfinite(f); }
SLANG_FORCE_INLINE bool F64_isinf(double f) { return SLANG_PRELUDE_STD isinf(f); }

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b) { return ::fmin(a, b); }
SLANG_FORCE_INLINE double F64_max(double a, double b) { return ::fmax(a, b); }
SLANG_FORCE_INLINE double F64_pow(double a, double b) { return ::pow(a, b); }
SLANG_FORCE_INLINE double F64_fmod(double a, double b) { return ::fmod(a, b); }
SLANG_FORCE_INLINE double F64_remainder(double a, double b) { return ::remainder(a, b); }
SLANG_FORCE_INLINE double F64_atan2(double a, double b) { return ::atan2(a, b); }

SLANG_FORCE_INLINE double F64_frexp(double x, int* e) { return ::frexp(x, e); }

SLANG_FORCE_INLINE double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c) { return ::fma(a, b, c); }

#endif // SLANG_LLVM

SLANG_FORCE_INLINE double F64_rsqrt(double f) { return 1.0 / F64_sqrt(f); }
SLANG_FORCE_INLINE double F64_sign(double f) { return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0); }
SLANG_FORCE_INLINE double F64_frac(double f) { return f - F64_floor(f); }

SLANG_FORCE_INLINE void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1 
	double a = radians * (1.0f /  (SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F64_floor(a);
    // Convert back to 0 - 2pi range
	return (a * (SLANG_PRELUDE_PI * 2));
}

// ----------------------------- I32 -----------------------------------------

SLANG_FORCE_INLINE int32_t I32_abs(int32_t f) { return (f < 0) ? -f : f; }

SLANG_FORCE_INLINE int32_t I32_min(int32_t a, int32_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE int32_t I32_max(int32_t a, int32_t b) { return a > b ? a : b; }

SLANG_FORCE_INLINE float I32_asfloat(int32_t x) { Union32 u; u.i = x; return u.f; }
SLANG_FORCE_INLINE uint32_t I32_asuint(int32_t x) { return uint32_t(x); }
SLANG_FORCE_INLINE double I32_asdouble(int32_t low, int32_t hi )
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

// ----------------------------- U32 -----------------------------------------

SLANG_FORCE_INLINE uint32_t U32_abs(uint32_t f) { return f; }

SLANG_FORCE_INLINE uint32_t U32_min(uint32_t a, uint32_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE uint32_t U32_max(uint32_t a, uint32_t b) { return a > b ? a : b; }

SLANG_FORCE_INLINE float U32_asfloat(uint32_t x) { Union32 u; u.u = x; return u.f; }
SLANG_FORCE_INLINE uint32_t U32_asint(int32_t x) { return uint32_t(x); } 

SLANG_FORCE_INLINE double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}


SLANG_FORCE_INLINE uint32_t U32_countbits(uint32_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return __builtin_popcount(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return __popcnt(v);
#else     
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE uint64_t U64_abs(uint64_t f) { return f; }

SLANG_FORCE_INLINE uint64_t U64_min(uint64_t a, uint64_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE uint64_t U64_max(uint64_t a, uint64_t b) { return a > b ? a : b; }

// TODO(JS): We don't define countbits for 64bit in stdlib currently.
// It's not clear from documentation if it should return 32 or 64 bits, if it exists. 
// 32 bits can always hold the result, and will be implicitly promoted. 
SLANG_FORCE_INLINE uint32_t U64_countbits(uint64_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)   
    return uint32_t(__builtin_popcountl(v));
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return uint32_t(__popcnt64(v));
#else     
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE int64_t I64_abs(int64_t f) { return (f < 0) ? -f : f; }

SLANG_FORCE_INLINE int64_t I64_min(int64_t a, int64_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE int64_t I64_max(int64_t a, int64_t b) { return a > b ? a : b; }


// ----------------------------- Interlocked ---------------------------------

#if SLANG_LLVM

#else // SLANG_LLVM

#   ifdef _WIN32
#       include <intrin.h>
#   endif

SLANG_FORCE_INLINE void InterlockedAdd(uint32_t* dest, uint32_t value, uint32_t* oldValue)
{
#   ifdef _WIN32
    *oldValue = _InterlockedExchangeAdd((long*)dest, (long)value);
#   else
    *oldValue = __sync_fetch_and_add(dest, value);
#   endif
}

#endif // SLANG_LLVM


// ----------------------- fmod --------------------------
SLANG_FORCE_INLINE float _slang_fmod(float x, float y)
{
    return F32_fmod(x, y);
}
SLANG_FORCE_INLINE double _slang_fmod(double x, double y)
{
    return F64_fmod(x, y);
}

#ifdef SLANG_PRELUDE_NAMESPACE
} 
#endif

#endif



static const int kSlangTorchTensorMaxDim = 5;

struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;
};


TensorView make_tensor_view(torch::Tensor val, const char* name, torch::ScalarType targetScalarType, bool requireContiguous)
{
    // We're currently not trying to implicitly cast or transfer to device for two reasons:
    // 1. There appears to be a bug with .to() where successive calls after the first one fail.
    // 2. Silent casts like this can cause large memory allocations & unexpected overheads. 
    //    It's better to be explicit.

    // Expect tensors to be on CUDA device
    if (!val.device().is_cuda())
        throw std::runtime_error(std::string(name).append(": tensor is not on CUDA device.").c_str());

    // Expect tensors to be the right type.
    if (val.dtype() != targetScalarType)
        throw std::runtime_error(std::string(name).append(": tensor is not of the expected type.").c_str());

    // Check that the tensor is contiguous
    if (requireContiguous && !val.is_contiguous())
        throw std::runtime_error(std::string(name).append(": tensor is not contiguous.").c_str());

    TensorView res = {};
    res.dimensionCount = val.dim();
    res.data = nullptr;
    size_t elementSize = 4;

    switch (val.scalar_type())
    {
    case torch::kInt8:
    case torch::kUInt8:
        elementSize = 1;
        res.data = (uint8_t*)val.data_ptr<uint8_t>();
        break;
    case torch::kBFloat16:
        elementSize = 2;
        res.data = (uint8_t*)val.data_ptr<torch::BFloat16>();
        break;
    case torch::kFloat16:
        elementSize = 2;
        res.data = (uint8_t*)val.data_ptr<at::Half>();
        break;
    case torch::kInt16:
        elementSize = 2;
        res.data = (uint8_t*)val.data_ptr<int16_t>();
        break;
    case torch::kFloat32:
        elementSize = 4;
        res.data = (uint8_t*)val.data_ptr<float>();
        break;
    case torch::kInt32:
        elementSize = 4;
        res.data = (uint8_t*)val.data_ptr<int32_t>();
        break;
    case torch::kFloat64:
        elementSize = 8;
        res.data = (uint8_t*)val.data_ptr<double>();
        break;
    case torch::kInt64:
        elementSize = 8;
        res.data = (uint8_t*)val.data_ptr<int64_t>();
        break;
    case torch::kBool:
        elementSize = 1;
        res.data = (uint8_t*)val.data_ptr<bool>();
        break;
    }

    if (val.dim() > kSlangTorchTensorMaxDim)
        throw std::runtime_error(std::string(name).append(": number of dimensions exceeds limit (").append(std::to_string(kSlangTorchTensorMaxDim)).append(")").c_str());

    bool isEmpty = true;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.strides[i] = val.stride(i) * elementSize;
        if (res.strides[i] == 0)
            throw std::runtime_error(std::string(name).append(": tensors with broadcasted dimensions are not supported (use tensor.contiguous() to make tensor whole)").c_str());

        res.sizes[i] = val.size(i);
        if (res.sizes[i] > 0)
            isEmpty = false;
    }

    if (!res.data && !isEmpty)
        throw std::runtime_error(std::string(name).append(": data pointer is invalid.").c_str());

    return res;
}

#define SLANG_PRELUDE_EXPORT

SLANG_PRELUDE_EXPORT
void __kernel__interpolate(TensorView _0, TensorView _1, TensorView _2, TensorView _3);

SLANG_PRELUDE_EXPORT
void __kernel__bake_uv(TensorView _0, TensorView _1, TensorView _2);

SLANG_PRELUDE_EXPORT
void interpolate(std::tuple<uint32_t, uint32_t, uint32_t> _blockSize_0, std::tuple<uint32_t, uint32_t, uint32_t> _gridSize_0, torch::Tensor attr_0, torch::Tensor indices_0, torch::Tensor rast_0, torch::Tensor output_0)
{
    Vector<uint32_t, 3>  _S1 = Vector<uint32_t, 3> (std::get<int(0)>(_blockSize_0), std::get<int(1)>(_blockSize_0), std::get<int(2)>(_blockSize_0));
    Vector<uint32_t, 3>  _S2 = Vector<uint32_t, 3> (std::get<int(0)>(_gridSize_0), std::get<int(1)>(_gridSize_0), std::get<int(2)>(_gridSize_0));
    TensorView _S3 = make_tensor_view(indices_0, "indices", torch::kInt32, true);
    TensorView _S4 = make_tensor_view(rast_0, "rast", torch::kFloat32, true);
    TensorView _S5 = make_tensor_view(output_0, "output", torch::kFloat32, true);
    TensorView _S6 = make_tensor_view(attr_0, "attr", torch::kFloat32, true);
    FixedArray<void *, 4>  _S7;
    _S7[int(0)] = &_S6;
    TensorView _S8 = _S3;
    _S7[int(1)] = &_S8;
    TensorView _S9 = _S4;
    _S7[int(2)] = &_S9;
    TensorView _S10 = _S5;
    _S7[int(3)] = &_S10;
    AT_CUDA_CHECK(cudaLaunchKernel((const void*)(__kernel__interpolate), slang_bit_cast<dim3>(_S2), slang_bit_cast<dim3>(_S1), &_S7[int(0)], 0, ((cudaStream_t)at::cuda::getCurrentCUDAStream())));
    return;
}

SLANG_PRELUDE_EXPORT
static std::tuple<std::tuple<const char*, const char*, const char*, const char*, const char*, const char*>, std::tuple<const char*, const char*, const char*, const char*>, const char*, const char*> __funcinfo__interpolate()
{
    return std::make_tuple(std::make_tuple(Slang::toTerminatedSlice("__blockSize").getBuffer(), Slang::toTerminatedSlice("__gridSize").getBuffer(), Slang::toTerminatedSlice("attr").getBuffer(), Slang::toTerminatedSlice("indices").getBuffer(), Slang::toTerminatedSlice("rast").getBuffer(), Slang::toTerminatedSlice("output").getBuffer()), std::make_tuple(Slang::toTerminatedSlice("").getBuffer(), Slang::toTerminatedSlice("").getBuffer(), Slang::toTerminatedSlice("").getBuffer(), Slang::toTerminatedSlice("").getBuffer()), Slang::toTerminatedSlice(""), Slang::toTerminatedSlice(""));
}

SLANG_PRELUDE_EXPORT
void bake_uv(std::tuple<uint32_t, uint32_t, uint32_t> _blockSize_1, std::tuple<uint32_t, uint32_t, uint32_t> _gridSize_1, torch::Tensor uv_0, torch::Tensor indices_1, torch::Tensor output_1)
{
    Vector<uint32_t, 3>  _S11 = Vector<uint32_t, 3> (std::get<int(0)>(_blockSize_1), std::get<int(1)>(_blockSize_1), std::get<int(2)>(_blockSize_1));
    Vector<uint32_t, 3>  _S12 = Vector<uint32_t, 3> (std::get<int(0)>(_gridSize_1), std::get<int(1)>(_gridSize_1), std::get<int(2)>(_gridSize_1));
    TensorView _S13 = make_tensor_view(indices_1, "indices", torch::kInt32, true);
    TensorView _S14 = make_tensor_view(output_1, "output", torch::kFloat32, true);
    TensorView _S15 = make_tensor_view(uv_0, "uv", torch::kFloat32, true);
    FixedArray<void *, 3>  _S16;
    _S16[int(0)] = &_S15;
    TensorView _S17 = _S13;
    _S16[int(1)] = &_S17;
    TensorView _S18 = _S14;
    _S16[int(2)] = &_S18;
    AT_CUDA_CHECK(cudaLaunchKernel((const void*)(__kernel__bake_uv), slang_bit_cast<dim3>(_S12), slang_bit_cast<dim3>(_S11), &_S16[int(0)], 0, ((cudaStream_t)at::cuda::getCurrentCUDAStream())));
    return;
}

SLANG_PRELUDE_EXPORT
static std::tuple<std::tuple<const char*, const char*, const char*, const char*, const char*>, std::tuple<const char*, const char*, const char*>, const char*, const char*> __funcinfo__bake_uv()
{
    return std::make_tuple(std::make_tuple(Slang::toTerminatedSlice("__blockSize").getBuffer(), Slang::toTerminatedSlice("__gridSize").getBuffer(), Slang::toTerminatedSlice("uv").getBuffer(), Slang::toTerminatedSlice("indices").getBuffer(), Slang::toTerminatedSlice("output").getBuffer()), std::make_tuple(Slang::toTerminatedSlice("").getBuffer(), Slang::toTerminatedSlice("").getBuffer(), Slang::toTerminatedSlice("").getBuffer()), Slang::toTerminatedSlice(""), Slang::toTerminatedSlice(""));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("interpolate", &interpolate, "interpolate");
    m.def("__funcinfo__interpolate", &__funcinfo__interpolate, "__funcinfo__interpolate");
    m.def("bake_uv", &bake_uv, "bake_uv");
    m.def("__funcinfo__bake_uv", &__funcinfo__bake_uv, "__funcinfo__bake_uv");
}
