/**
 * MedCHR.ai — Shared JavaScript Utilities
 * Loaded on every page via base.html.
 */

// ── Toast Notifications ─────────────────────────────────────
function showToast(message, type) {
  type = type || 'info';
  var container = document.getElementById('toastContainer');
  if (!container) return;
  var toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  var iconSpan = document.createElement('span');
  var icon = type === 'success' ? '\u2713' : type === 'error' ? '\u2717' : '\u2139';
  iconSpan.textContent = icon;
  var msgSpan = document.createElement('span');
  msgSpan.textContent = message;
  toast.appendChild(iconSpan);
  toast.appendChild(msgSpan);
  container.appendChild(toast);
  setTimeout(function () { toast.remove(); }, 4000);
}

// ── Fetch Wrapper with Error Toast ──────────────────────────
function apiFetch(url, options) {
  options = options || {};
  return fetch(url, Object.assign({ credentials: 'same-origin' }, options))
    .then(function (res) {
      if (!res.ok) {
        throw new Error('Request failed: ' + res.status);
      }
      return res;
    })
    .catch(function (err) {
      showToast(err.message || 'Something went wrong', 'error');
      throw err;
    });
}

// ── Form Loading State ──────────────────────────────────────
document.querySelectorAll('form').forEach(function (form) {
  form.addEventListener('submit', function () {
    var btn = form.querySelector('button[type="submit"]');
    if (btn && !btn.disabled) {
      btn.disabled = true;
      btn.dataset.originalText = btn.textContent;
      var spinner = document.createElement('span');
      spinner.className = 'spinner';
      btn.textContent = '';
      btn.appendChild(spinner);
      btn.appendChild(document.createTextNode(' Processing\u2026'));
    }
  });
});

// ── Collapsible Sections ────────────────────────────────────
document.querySelectorAll('.collapsible-header').forEach(function (header) {
  header.setAttribute('role', 'button');
  header.setAttribute('tabindex', '0');
  header.addEventListener('click', toggleCollapsible);
  header.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggleCollapsible.call(this);
    }
  });
});

function toggleCollapsible() {
  var content = this.nextElementSibling;
  var chevron = this.querySelector('.chevron');
  var isOpen = content.classList.toggle('open');
  this.setAttribute('aria-expanded', isOpen);
  if (chevron) chevron.classList.toggle('open');
}

// ── Mobile Sidebar Toggle ───────────────────────────────────
(function () {
  var mobileMenuBtn = document.getElementById('mobileMenuBtn');
  var sidebar = document.getElementById('sidebar');
  if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener('click', function () {
      var isOpen = sidebar.classList.toggle('open');
      mobileMenuBtn.setAttribute('aria-expanded', isOpen);
    });
  }
})();

// ── Session Timeout ─────────────────────────────────────────
(function () {
  var extendSessionBtn = document.getElementById('extendSessionBtn');
  if (extendSessionBtn) {
    extendSessionBtn.addEventListener('click', extendSession);
  }

  var sessionTimeoutWarning = null;
  var sessionTimeoutLogout = null;
  var countdownInterval = null;
  var WARNING_BEFORE_SECONDS = 60;
  var SESSION_TIMEOUT_SECONDS = 900;

  function resetSessionTimer() {
    clearTimeout(sessionTimeoutWarning);
    clearTimeout(sessionTimeoutLogout);
    clearInterval(countdownInterval);
    var modal = document.getElementById('sessionTimeoutModal');
    if (modal) modal.style.display = 'none';
    sessionTimeoutWarning = setTimeout(showTimeoutWarning, (SESSION_TIMEOUT_SECONDS - WARNING_BEFORE_SECONDS) * 1000);
    sessionTimeoutLogout = setTimeout(function () {
      window.location.href = '/ui/logout?timeout=1';
    }, SESSION_TIMEOUT_SECONDS * 1000);
  }

  function showTimeoutWarning() {
    var modal = document.getElementById('sessionTimeoutModal');
    var countdown = document.getElementById('timeoutCountdown');
    if (!modal) return;
    modal.style.display = 'flex';
    var remaining = WARNING_BEFORE_SECONDS;
    countdown.textContent = remaining;
    countdownInterval = setInterval(function () {
      remaining--;
      countdown.textContent = remaining;
      if (remaining <= 0) clearInterval(countdownInterval);
    }, 1000);
  }

  function extendSession() {
    fetch('/health', { method: 'GET', credentials: 'same-origin' })
      .then(function () { resetSessionTimer(); })
      .catch(function () { window.location.href = '/ui/login'; });
  }

  ['mousedown', 'keydown', 'scroll', 'touchstart'].forEach(function (event) {
    document.addEventListener(event, resetSessionTimer, { passive: true });
  });

  resetSessionTimer();

  if (window.location.search.indexOf('timeout=1') !== -1) {
    showToast('Session expired due to inactivity', 'warning');
  }

  // Expose for other scripts
  window.resetSessionTimer = resetSessionTimer;
})();

// ── Sidebar Active State ────────────────────────────────────
(function () {
  function updateSidebarActiveLink() {
    var hash = window.location.hash;
    var subLinks = document.querySelectorAll('.sidebar-sub-link');
    if (!subLinks.length) return;

    if (hash || window.location.pathname.match(/\/ui\/patients\/[^\/]+$/)) {
      subLinks.forEach(function (link) {
        link.classList.remove('active');
        var linkHref = link.getAttribute('href');
        if (hash && linkHref.indexOf(hash) !== -1) {
          link.classList.add('active');
        } else if (!hash && linkHref.indexOf('#') === -1 && linkHref.indexOf('report') === -1) {
          link.classList.add('active');
        }
      });
    }
  }

  window.addEventListener('hashchange', updateSidebarActiveLink);
  window.addEventListener('DOMContentLoaded', updateSidebarActiveLink);
})();
