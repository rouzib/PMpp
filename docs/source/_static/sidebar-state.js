(() => {
  "use strict"

  const nav = document.querySelector(".bd-sidebar-primary nav.bd-docs-nav")
  if (!nav) return

  const storageKey = "pmpp-sidebar-tree-state-v1"
  const hasOwn = (object, key) =>
    Object.prototype.hasOwnProperty.call(object, key)

  const readState = () => {
    try {
      const parsed = JSON.parse(window.localStorage.getItem(storageKey) || "{}")
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? parsed
        : {}
    } catch {
      return {}
    }
  }

  const state = readState()

  const writeState = () => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(state))
    } catch {
      // Navigation still works when storage is disabled or unavailable.
    }
  }

  const directChild = (element, selector) =>
    Array.from(element.children).find((child) => child.matches(selector))

  const normalizedPath = (href) => {
    try {
      const url = new URL(href, document.baseURI)
      let path = url.pathname
      if (path.endsWith("/index.html")) {
        path = path.slice(0, -"index.html".length)
      }
      return path.replace(/\/+$/, "") || "/"
    } catch {
      return null
    }
  }

  const branches = Array.from(
    nav.querySelectorAll("li.has-children > details"),
  )
    .map((details) => {
      const item = details.parentElement
      const link = directChild(item, "a.reference.internal")
      const key = link ? normalizedPath(link.getAttribute("href")) : null
      return key ? { details, item, key, link } : null
    })
    .filter(Boolean)

  const branchByItem = new Map(branches.map((branch) => [branch.item, branch]))

  for (const branch of branches) {
    if (hasOwn(state, branch.key) && typeof state[branch.key] === "boolean") {
      branch.details.open = state[branch.key]
    }
  }

  const openBranchAndAncestors = (branch) => {
    let item = branch.item
    while (item && nav.contains(item)) {
      const current = branchByItem.get(item)
      if (current) {
        current.details.open = true
        state[current.key] = true
      }
      item = item.parentElement?.closest("li.has-children") || null
    }
    writeState()
  }

  for (const branch of branches) {
    branch.details.addEventListener("toggle", () => {
      state[branch.key] = branch.details.open
      writeState()
    })

    branch.link.addEventListener("click", () => {
      openBranchAndAncestors(branch)
    })
  }
})()
