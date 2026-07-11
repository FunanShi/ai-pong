// Adapted from a GL presentation shim I wrote for an earlier project, reused with permission.
// Dev-machine in-container GLX->GLXWindow present workaround; imgui-only fallback in ui_pong.
//
// LD_PRELOAD workaround for a host-specific display bug (dev machine only).
//
// On this machine (RTX 4070 SUPER, driver 595.71.05-open, Xorg+mutter),
// glXSwapBuffers from inside the container silently drops frames when the
// drawable is a GLX 1.3 GLXWindow (created via glXCreateWindow): swaps return
// immediately, unthrottled, and nothing reaches the screen. Swapping on the
// bare X window works (verified with tools/glxtest.c modes L/F/FW). GLFW —
// and therefore the MuJoCo viewer in librobot_lib — always swaps on a
// GLXWindow, so the simulator window rendered black.
//
// The shim makes glXCreateWindow return the bare X window (a valid
// GLXDrawable for MakeCurrent/SwapBuffers) and turns glXDestroyWindow into a
// no-op for it. GLFW looks these functions up with dlsym() on its own
// dlopened libGL handle, which bypasses ordinary symbol interposition, so we
// interpose dlsym itself.
//
// Use with __GL_SYNC_TO_VBLANK=0 (the synchronized flip path from container
// clients is also broken on this setup; vsync-off swaps take the working
// blit path).
//
// Build:  cc -shared -fPIC glx_present_shim.c -o glx_present_shim.so
// Use:    LD_PRELOAD=/path/to/glx_present_shim.so __GL_SYNC_TO_VBLANK=0 <app>
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef unsigned long XID;

static XID shim_glXCreateWindow(void *dpy, void *config, XID win,
                                const int *attribs) {
  (void)dpy; (void)config; (void)attribs;
  fprintf(stderr, "[glx_present_shim] glXCreateWindow -> bare window 0x%lx\n",
          win);
  return win;
}

static void shim_glXDestroyWindow(void *dpy, XID window) {
  (void)dpy; (void)window;
}

typedef void *(*dlsym_fn)(void *, const char *);

static dlsym_fn real_dlsym(void) {
  static dlsym_fn real = NULL;
  if (!real)
    real = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  return real;
}

void *dlsym(void *handle, const char *name) {
  if (name) {
    if (strcmp(name, "glXCreateWindow") == 0)
      return (void *)shim_glXCreateWindow;
    if (strcmp(name, "glXDestroyWindow") == 0)
      return (void *)shim_glXDestroyWindow;
  }
  return real_dlsym()(handle, name);
}

// Also export directly for apps that link/lookup normally.
XID glXCreateWindow(void *dpy, void *config, XID win, const int *attribs) {
  return shim_glXCreateWindow(dpy, config, win, attribs);
}
void glXDestroyWindow(void *dpy, XID window) {
  shim_glXDestroyWindow(dpy, window);
}
