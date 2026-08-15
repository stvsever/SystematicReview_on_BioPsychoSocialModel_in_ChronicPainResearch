(function () {
  "use strict";

  const data = window.BPS_GRAPH_DATA;
  if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
    document.body.textContent = "The graph data file could not be loaded.";
    return;
  }

  const canvas = document.getElementById("graphCanvas");
  const context = canvas.getContext("2d", { alpha: false });
  const stage = document.querySelector(".graph-stage");
  const tooltip = document.getElementById("tooltip");
  const quickCard = document.getElementById("quickCard");
  const detailPanel = document.getElementById("detailPanel");
  const detailContent = document.getElementById("detailContent");
  const zoomReadout = document.getElementById("zoomReadout");
  const searchInput = document.getElementById("graphSearch");
  const nodes = data.nodes.map((node) => ({
    ...node,
    x: 0,
    y: 0,
    targetX: 0,
    targetY: 0,
    vx: 0,
    vy: 0,
    fx: null,
    fy: null,
    visible: false,
    searchMatch: false,
    manualOffsetX: 0,
    manualOffsetY: 0,
  }));
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const edges = data.edges.map((edge) => ({
    ...edge,
    sourceNode: nodeById.get(edge.source),
    targetNode: nodeById.get(edge.target),
  }));
  const parents = new Map();
  const children = new Map();
  edges.forEach((edge) => {
    parents.set(edge.target, edge.source);
    if (!children.has(edge.source)) children.set(edge.source, []);
    children.get(edge.source).push(edge.target);
  });

  const allFieldIds = data.filters.field_groups.flatMap((group) => group.fields.map((field) => field.id));
  const state = {
    selectedArticles: new Set(data.filters.articles.map((item) => item.id)),
    selectedProviders: new Set(data.filters.providers.map((item) => item.id)),
    selectedFields: new Set(allFieldIds),
    activeField: null,
    activeArticle: null,
    activeProvider: null,
    showAll: false,
    popupEnabled: true,
    search: "",
    searchTruncated: false,
    scale: 0.6,
    offsetX: 0,
    offsetY: 0,
    selectedNode: null,
    hoveredNode: null,
    draggingNode: null,
    draggingCluster: [],
    draggingLastWorld: null,
    draggingWasPinned: false,
    draggingAnchorCaptured: false,
    panning: false,
    pointerStart: null,
    showLabels: true,
    quickCardDragging: false,
    quickCardDragStart: null,
    settledFrames: 0,
  };

  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let lastTime = performance.now();
  let lastDrawTime = 0;
  let visibleNodes = [];
  let visibleEdges = [];
  let visibleNodesDrawOrder = [];
  const MIN_SCALE = 0.008;
  const MAX_SCALE = 10;

  function inheritedManualOffset(node) {
    let offsetX = 0;
    let offsetY = 0;
    let cursor = node;
    while (cursor) {
      offsetX += cursor.manualOffsetX;
      offsetY += cursor.manualOffsetY;
      cursor = parents.has(cursor.id) ? nodeById.get(parents.get(cursor.id)) : null;
    }
    return { x: offsetX, y: offsetY };
  }

  function positionNode(node, x, y) {
    const manualOffset = inheritedManualOffset(node);
    const positionedX = x + manualOffset.x;
    const positionedY = y + manualOffset.y;
    node.targetX = positionedX;
    node.targetY = positionedY;
    node.x = positionedX;
    node.y = positionedY;
    node.vx = 0;
    node.vy = 0;
    if (node.fx !== null) {
      node.fx = positionedX;
      node.fy = positionedY;
    }
  }

  function packedOrbitSlot(index, total, baseRadius, minimumSpacing, startAngle) {
    let ringStart = 0;
    let radius = baseRadius;
    let ring = 0;
    while (ringStart < total) {
      const capacity = Math.max(6, Math.floor(Math.PI * 2 * radius / minimumSpacing));
      const ringCount = Math.min(capacity, total - ringStart);
      if (index < ringStart + ringCount) {
        const ringIndex = index - ringStart;
        return {
          angle: startAngle + ring * 0.19 + Math.PI * 2 * ringIndex / Math.max(1, ringCount),
          radius,
        };
      }
      ringStart += ringCount;
      radius += minimumSpacing * 1.05;
      ring += 1;
    }
    return { angle: startAngle, radius };
  }

  function initializePositions() {
    const root = nodes.find((node) => node.type === "run");
    if (!root) return;
    positionNode(root, 0, 0);

    // A group holds coding fields directly, or holds entities that hold them. The
    // sector each group gets is proportional to its coding fields however deep
    // they sit, and an entity node is parked on the mean angle of its own fields,
    // between the group ring and the field ring.
    function branchesOf(node) {
      return (children.get(node.id) || []).map((id) => nodeById.get(id)).filter(Boolean);
    }

    // Headings nest to any depth: a group can hold entities, and an entity can
    // hold kinds of its own. The coding fields are the leaves, wherever they sit.
    function leafFieldsOf(node, depth, depths) {
      const leaves = [];
      branchesOf(node).forEach((branch) => {
        if (branch.type === "subgroup") {
          depths.set(branch.id, depth + 1);
          leaves.push(...leafFieldsOf(branch, depth + 1, depths));
        } else {
          leaves.push(branch);
        }
      });
      return leaves;
    }

    const groupNodes = branchesOf(root);
    const headingDepth = new Map();
    const leavesByGroup = new Map(
      groupNodes.map((group) => [group.id, leafFieldsOf(group, 0, headingDepth)]));
    const maxHeadingDepth = Math.max(1, ...headingDepth.values());
    const totalFields = Math.max(1, groupNodes.reduce(
      (total, group) => total + leavesByGroup.get(group.id).length, 0));
    let angleCursor = -Math.PI / 2;
    groupNodes.forEach((group) => {
      const leaves = leavesByGroup.get(group.id);
      const sector = Math.PI * 2 * leaves.length / totalFields;
      const centerAngle = angleCursor + sector / 2;
      group.layoutAngle = centerAngle;
      const groupRadius = state.showAll ? 1800 : 330;
      const groupX = Math.cos(centerAngle) * groupRadius;
      const groupY = Math.sin(centerAngle) * groupRadius;
      positionNode(group, groupX, groupY);
      const padding = Math.min(0.06, sector * 0.12);
      const subgroupAngles = new Map();

      leaves.forEach((field, fieldIndex) => {
        const ratio = (fieldIndex + 0.5) / Math.max(1, leaves.length);
        const angle = angleCursor + padding + ratio * Math.max(0.01, sector - padding * 2);
        const radius = state.showAll ? 5000 : 760 + (fieldIndex % 2) * 86;
        field.layoutAngle = angle;
        const fieldX = Math.cos(angle) * radius;
        const fieldY = Math.sin(angle) * radius;
        positionNode(field, fieldX, fieldY);
        // Every heading above this field takes the field's angle into account, so
        // a heading ends up on the mean bearing of its whole subtree.
        let ancestorId = parents.get(field.id);
        while (ancestorId) {
          const ancestor = nodeById.get(ancestorId);
          if (!ancestor || ancestor.type !== "subgroup") break;
          const seen = subgroupAngles.get(ancestor.id) || [];
          seen.push(angle);
          subgroupAngles.set(ancestor.id, seen);
          ancestorId = parents.get(ancestor.id);
        }

        const providerNodes = (children.get(field.id) || []).map((id) => nodeById.get(id)).filter(Boolean);
        const selectedProviderNodes = providerNodes.filter((provider) => state.selectedProviders.has(provider.provider));
        providerNodes.forEach((provider, providerIndex) => {
          const selectedIndex = Math.max(0, selectedProviderNodes.indexOf(provider));
          const providerAngle = angle + Math.PI * 2 * selectedIndex / Math.max(1, selectedProviderNodes.length);
          const providerOrbit = state.showAll ? 145 : 210;
          const omitProviderHub = selectedProviderNodes.length === 1 && selectedProviderNodes[0] === provider;
          const providerX = omitProviderHub ? fieldX : fieldX + Math.cos(providerAngle) * providerOrbit;
          const providerY = omitProviderHub ? fieldY : fieldY + Math.sin(providerAngle) * providerOrbit;
          provider.orbitRadius = omitProviderHub ? 0 : providerOrbit;
          positionNode(provider, providerX, providerY);

          const articleNodes = (children.get(provider.id) || []).map((id) => nodeById.get(id)).filter(Boolean);
          articleNodes.forEach((article, articleIndex) => {
            const articleSlot = packedOrbitSlot(
              articleIndex,
              articleNodes.length,
              omitProviderHub ? 255 : state.showAll ? 58 : 88,
              state.showAll ? 23 : 34,
              providerAngle,
            );
            const articleAngle = articleSlot.angle;
            const articleOrbit = articleSlot.radius;
            article.orbitRadius = articleOrbit;
            const articleX = providerX + Math.cos(articleAngle) * articleOrbit;
            const articleY = providerY + Math.sin(articleAngle) * articleOrbit;
            positionNode(article, articleX, articleY);

            const itemNodes = (children.get(article.id) || []).map((id) => nodeById.get(id)).filter(Boolean);
            itemNodes.forEach((item, itemIndex) => {
              const itemSlot = packedOrbitSlot(
                itemIndex,
                itemNodes.length,
                state.activeArticle === article.id ? 34 : 24,
                state.activeArticle === article.id ? 18 : 16,
                articleAngle + 0.31,
              );
              item.orbitRadius = itemSlot.radius;
              positionNode(
                item,
                articleX + Math.cos(itemSlot.angle) * itemSlot.radius,
                articleY + Math.sin(itemSlot.angle) * itemSlot.radius,
              );
            });
          });
        });
      });

      subgroupAngles.forEach((angles, subgroupId) => {
        const subgroup = nodeById.get(subgroupId);
        if (!subgroup) return;
        const meanAngle = angles.reduce((total, value) => total + value, 0) / angles.length;
        // Heading rings sit evenly between the group ring and the field ring, one
        // ring per nesting level, so depth in the tree reads as distance out.
        const innerRadius = state.showAll ? 1800 : 330;
        const outerRadius = state.showAll ? 5000 : 760;
        const depth = headingDepth.get(subgroupId) || 1;
        const subgroupRadius = innerRadius
          + (outerRadius - innerRadius) * depth / (maxHeadingDepth + 1);
        subgroup.layoutAngle = meanAngle;
        positionNode(
          subgroup,
          Math.cos(meanAngle) * subgroupRadius,
          Math.sin(meanAngle) * subgroupRadius,
        );
      });
      angleCursor += sector;
    });
  }

  function checkboxOption(value, label, meta, groupName, swatch) {
    const wrapper = document.createElement("label");
    wrapper.className = "filter-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    input.checked = true;
    input.dataset.filterGroup = groupName;
    const check = document.createElement("span");
    check.className = "check-ui";
    const copy = document.createElement("span");
    copy.className = "filter-label";
    copy.textContent = label;
    const trailing = document.createElement("span");
    if (swatch) {
      trailing.className = "swatch";
      trailing.style.color = swatch;
      trailing.style.background = swatch;
    } else {
      trailing.className = "filter-meta";
      trailing.textContent = meta || "";
    }
    input.addEventListener("change", updateFiltersFromControls);
    wrapper.append(input, check, copy, trailing);
    return wrapper;
  }

  function buildFilters() {
    const articleHost = document.getElementById("articleFilters");
    data.filters.articles.forEach((article) => {
      articleHost.appendChild(checkboxOption(article.id, `${article.id} · ${article.label}`, "", "articles", article.color));
    });

    const providerHost = document.getElementById("providerFilters");
    data.filters.providers.forEach((provider) => {
      providerHost.appendChild(checkboxOption(provider.id, `${provider.label} · ${provider.provider}`, "", "providers", provider.color));
    });

    const fieldsHost = document.getElementById("fieldFilters");
    data.filters.field_groups.forEach((group, groupIndex) => {
      const details = document.createElement("details");
      details.className = "field-group";
      details.open = groupIndex < 3;
      const summary = document.createElement("summary");
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.color = group.color;
      swatch.style.background = group.color;
      const title = document.createElement("span");
      title.textContent = group.name;
      summary.append(swatch, title);
      const fields = document.createElement("div");
      fields.className = "group-fields";
      group.fields.forEach((field) => {
        fields.appendChild(checkboxOption(field.id, field.label, "", "fields", field.color || group.color));
      });
      details.append(summary, fields);
      fieldsHost.appendChild(details);
    });

    document.querySelectorAll(".mini-actions button").forEach((button) => {
      button.addEventListener("click", () => {
        const checked = button.dataset.action === "all";
        document.querySelectorAll(`input[data-filter-group="${button.dataset.set}"]`).forEach((input) => {
          input.checked = checked;
        });
        updateFiltersFromControls();
      });
    });
  }

  function selectedValues(groupName) {
    return new Set(Array.from(document.querySelectorAll(`input[data-filter-group="${groupName}"]:checked`)).map((input) => input.value));
  }

  function ancestorIds(nodeId) {
    const result = new Set();
    let cursor = nodeId;
    while (parents.has(cursor)) {
      cursor = parents.get(cursor);
      result.add(cursor);
    }
    return result;
  }

  function passesFilters(node) {
    if (node.field && !state.selectedFields.has(node.field)) return false;
    if (["article", "item"].includes(node.type) && node.article_id && !state.selectedArticles.has(node.article_id)) return false;
    if (["provider", "article", "item"].includes(node.type) && node.provider && !state.selectedProviders.has(node.provider)) return false;
    if (node.type === "group" || node.type === "subgroup") {
      // A heading survives while any coding field beneath it, at any depth, is on.
      const stack = [...(children.get(node.id) || [])];
      while (stack.length) {
        const child = nodeById.get(stack.pop());
        if (!child) continue;
        if (child.type === "field") {
          if (state.selectedFields.has(child.field)) return true;
        } else if (child.type === "subgroup") {
          stack.push(...(children.get(child.id) || []));
        }
      }
      return false;
    }
    return true;
  }

  function baseVisibility(node) {
    if (node.type === "run") return true;
    if (node.type === "provider" && state.selectedProviders.size === 1) return false;
    if (state.showAll) return true;
    if (node.type === "group" || node.type === "subgroup" || node.type === "field") return true;
    if (!state.activeField) return false;
    const field = nodeById.get(state.activeField);
    if (!field) return false;
    if (node.type === "provider" || node.type === "article") return node.field === field.field;
    if (node.type === "item") return parents.get(node.id) === state.activeArticle;
    return false;
  }

  function updateVisibility() {
    const query = state.search.trim().toLowerCase();
    const searchMatches = new Set();
    const searchContext = new Set();
    state.searchTruncated = false;
    nodes.forEach((node) => { node.searchMatch = false; });
    if (query) {
      const matches = nodes.filter((node) => passesFilters(node) && node.search.includes(query));
      state.searchTruncated = matches.length > 240;
      matches.slice(0, 240).forEach((node) => {
        node.searchMatch = true;
        searchMatches.add(node.id);
        ancestorIds(node.id).forEach((id) => searchContext.add(id));
      });
    }

    nodes.forEach((node) => {
      const filterPass = passesFilters(node);
      const structurePass = query
        ? searchMatches.has(node.id) || searchContext.has(node.id)
        : baseVisibility(node);
      const redundantProviderHub = node.type === "provider" && state.selectedProviders.size === 1;
      node.visible = filterPass && structurePass && !redundantProviderHub;
    });
    visibleNodes = nodes.filter((node) => node.visible);
    visibleEdges = edges.filter((edge) => edge.sourceNode.visible && edge.targetNode.visible);
    if (state.selectedProviders.size === 1) {
      visibleNodes.filter((node) => node.type === "field").forEach((field) => {
        (children.get(field.id) || []).forEach((providerId) => {
          const provider = nodeById.get(providerId);
          if (!provider || !state.selectedProviders.has(provider.provider)) return;
          (children.get(provider.id) || []).forEach((articleId) => {
            const article = nodeById.get(articleId);
            if (!article || !article.visible) return;
            visibleEdges.push({
              source: field.id,
              target: article.id,
              kind: "single_provider_article",
              sourceNode: field,
              targetNode: article,
            });
          });
        });
      });
    }
    visibleNodesDrawOrder = [...visibleNodes].sort((a, b) => b.level - a.level);
    if (state.selectedNode && !state.selectedNode.visible) clearDetail();
    if (state.hoveredNode && !state.hoveredNode.visible) {
      state.hoveredNode = null;
      tooltip.style.display = "none";
    }
    state.settledFrames = 0;
    updateStatus();
    updateViewTrail();
  }

  function updateFiltersFromControls() {
    state.selectedArticles = selectedValues("articles");
    state.selectedProviders = selectedValues("providers");
    state.selectedFields = selectedValues("fields");
    if (state.activeField) {
      const fieldNode = nodeById.get(state.activeField);
      if (!fieldNode || !state.selectedFields.has(fieldNode.field)) {
        state.activeField = null;
        state.activeArticle = null;
        state.activeProvider = null;
      }
    }
    if (state.activeArticle) {
      const articleNode = nodeById.get(state.activeArticle);
      if (!articleNode || !state.selectedArticles.has(articleNode.article_id) || !state.selectedProviders.has(articleNode.provider)) {
        state.activeArticle = null;
      }
    }
    if (state.activeProvider) {
      const providerNode = nodeById.get(state.activeProvider);
      if (!providerNode || !state.selectedProviders.has(providerNode.provider)) state.activeProvider = null;
    }
    if (state.selectedProviders.size === 1) state.activeProvider = null;
    initializePositions();
    updateVisibility();
  }

  function updateStatus() {
    const suffix = state.searchTruncated ? " · first 240 matches" : "";
    document.getElementById("visibleStatus").textContent = `${visibleNodes.length.toLocaleString()} nodes · ${visibleEdges.length.toLocaleString()} links visible${suffix}`;
  }

  function updateViewTrail() {
    const host = document.getElementById("viewTrail");
    if (state.search.trim()) {
      host.textContent = `Search results · ${state.search.trim()}`;
      return;
    }
    if (state.showAll) {
      host.textContent = "Complete graph · every selected field, article, provider, and extracted item";
      return;
    }
    if (!state.activeField) {
      host.textContent = "Scheme overview · field groups and coding fields";
      return;
    }
    const field = nodeById.get(state.activeField);
    if (state.activeArticle) {
      const article = nodeById.get(state.activeArticle);
      host.textContent = `${field.field_group} / ${field.label} / ${article.provider} / ${article.article_id} / extracted items`;
    } else if (state.activeProvider) {
      const provider = nodeById.get(state.activeProvider);
      host.textContent = `${field.field_group} / ${field.label} / ${provider.provider} / papers`;
    } else {
      host.textContent = `${field.field_group} / ${field.label} / providers and papers`;
    }
  }

  function buildHeadline() {
    const values = [
      [data.meta.n_papers, "papers"],
      [data.meta.n_providers, "providers"],
      [data.meta.n_codings, "codings"],
      [allFieldIds.length, "fields"],
    ];
    const host = document.getElementById("headlineStats");
    values.forEach(([value, label]) => {
      const chip = document.createElement("div");
      chip.className = "stat-chip";
      const strong = document.createElement("strong");
      strong.textContent = Number(value).toLocaleString();
      const span = document.createElement("span");
      span.textContent = label;
      chip.append(strong, span);
      host.appendChild(chip);
    });
  }

  function resizeCanvas() {
    const bounds = stage.getBoundingClientRect();
    width = Math.max(1, bounds.width);
    height = Math.max(1, bounds.height);
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(width * pixelRatio);
    canvas.height = Math.floor(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }

  function worldToScreen(x, y) {
    return { x: width / 2 + state.offsetX + x * state.scale, y: height / 2 + state.offsetY + y * state.scale };
  }

  function screenToWorld(x, y) {
    return { x: (x - width / 2 - state.offsetX) / state.scale, y: (y - height / 2 - state.offsetY) / state.scale };
  }

  function hexToRgb(hex) {
    const normalized = hex.replace("#", "");
    const value = parseInt(normalized.length === 3 ? normalized.split("").map((character) => character + character).join("") : normalized, 16);
    return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
  }

  function rgba(hex, alpha) {
    const color = hexToRgb(hex);
    return `rgba(${color.r},${color.g},${color.b},${alpha})`;
  }

  function canvasPalette() {
    const light = document.documentElement.dataset.theme === "light";
    return light
      ? { background: "#f5f7fa", edge: "78, 92, 116", dimmedEdgeFactor: 0.42, label: "#26344a", labelMuted: "#526078", labelBackground: "rgba(255,255,255,0.84)" }
      : { background: "#090c13", edge: "133, 151, 181", dimmedEdgeFactor: 0.5, label: "#eaf1ff", labelMuted: "#b3bfd2", labelBackground: "rgba(7,10,16,0.78)" };
  }

  function simulate(delta) {
    if (state.settledFrames > 260 && !state.draggingNode) return;
    if (visibleNodes.length > 3500 && !state.draggingNode) return;
    const step = Math.min(1.35, delta / 16.67);

    visibleNodes.forEach((node) => {
      if (node.fx !== null) return;
      const activeLeaf = node.type === "item" && parents.get(node.id) === state.activeArticle;
      const strength = node.type === "run" ? 0.12 : node.type === "group" ? 0.045 : node.type === "subgroup" ? 0.05 : node.type === "field" ? 0.06 : activeLeaf ? 0.075 : 0.026;
      node.vx += (node.targetX - node.x) * strength * step;
      node.vy += (node.targetY - node.y) * strength * step;
    });

    visibleEdges.forEach((edge) => {
      const source = edge.sourceNode;
      const target = edge.targetNode;
      let dx = target.x - source.x;
      let dy = target.y - source.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const targetDistance = Math.max(20, Math.hypot(target.targetX - source.targetX, target.targetY - source.targetY));
      const force = (distance - targetDistance) * 0.0018 * step;
      dx /= distance;
      dy /= distance;
      if (source.fx === null) { source.vx += dx * force; source.vy += dy * force; }
      if (target.fx === null) { target.vx -= dx * force; target.vy -= dy * force; }
    });

    const buckets = new Map();
    const cell = 92;
    visibleNodes.forEach((node) => {
      const key = `${Math.floor(node.x / cell)},${Math.floor(node.y / cell)}`;
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(node);
    });
    visibleNodes.forEach((node) => {
      const bx = Math.floor(node.x / cell);
      const by = Math.floor(node.y / cell);
      for (let gridX = bx - 1; gridX <= bx + 1; gridX += 1) {
        for (let gridY = by - 1; gridY <= by + 1; gridY += 1) {
          (buckets.get(`${gridX},${gridY}`) || []).forEach((other) => {
            if (other === node || other.id < node.id) return;
            let dx = other.x - node.x;
            let dy = other.y - node.y;
            let distance = Math.hypot(dx, dy);
            if (distance < 0.1) {
              const angle = (hash(node.id + other.id) % 628) / 100;
              dx = Math.cos(angle);
              dy = Math.sin(angle);
              distance = 1;
            }
            const minimum = node.size + other.size + (isSchemeOverviewNode(node) || isSchemeOverviewNode(other) ? 20 : 13);
            const influence = minimum * 2.7;
            if (distance >= influence) return;
            const push = Math.max(0, (influence - distance) / influence) * (distance < minimum ? 0.48 : 0.08) * step;
            dx /= distance;
            dy /= distance;
            if (node.fx === null) { node.vx -= dx * push; node.vy -= dy * push; }
            if (other.fx === null) { other.vx += dx * push; other.vy += dy * push; }
          });
        }
      }
    });

    let energy = 0;
    visibleNodes.forEach((node) => {
      if (node.fx !== null) {
        node.x = node.fx;
        node.y = node.fy;
        node.vx = 0;
        node.vy = 0;
        return;
      }
      node.vx *= 0.74;
      node.vy *= 0.74;
      node.x += node.vx * step;
      node.y += node.vy * step;
      energy += Math.abs(node.vx) + Math.abs(node.vy);
    });
    state.settledFrames += energy < 0.04 ? 4 : 1;
  }

  function connectedSet(node) {
    if (!node) return new Set();
    const result = new Set();
    let cursor = node;
    while (cursor) {
      result.add(cursor.id);
      cursor = parents.has(cursor.id) ? nodeById.get(parents.get(cursor.id)) : null;
    }
    if (node.type !== "item") {
      (children.get(node.id) || []).forEach((id) => {
        const child = nodeById.get(id);
        if (child && child.visible) result.add(id);
        if (child && !child.visible) {
          (children.get(child.id) || []).forEach((grandchildId) => {
            const grandchild = nodeById.get(grandchildId);
            if (grandchild && grandchild.visible) result.add(grandchild.id);
          });
        }
      });
    }
    return result;
  }

  function focusedPathSet(node) {
    const result = new Set();
    let cursor = node;
    while (cursor) {
      result.add(cursor.id);
      cursor = parents.has(cursor.id) ? nodeById.get(parents.get(cursor.id)) : null;
    }
    return result;
  }

  function intersects(box, other) {
    return !(box.right < other.left || box.left > other.right || box.bottom < other.top || box.top > other.bottom);
  }

  function truncateLabel(label, maxWidth) {
    let text = label;
    while (context.measureText(text).width > maxWidth && text.length > 12) text = text.slice(0, -2);
    return text === label ? text : `${text.trimEnd()}...`;
  }

  function graphDisplayLabel(node) {
    return node.type === "run" ? "CODING SCHEME" : node.label;
  }

  const OVERVIEW_TYPES = new Set(["run", "group", "subgroup", "field"]);
  const HEADING_TYPES = new Set(["run", "group", "subgroup"]);

  function isSchemeOverviewNode(node) {
    return OVERVIEW_TYPES.has(node.type);
  }

  function drawLabels(palette, focusSet) {
    if (!state.showLabels) return;
    const focusNode = state.selectedNode || state.hoveredNode;
    const pathSet = focusedPathSet(focusNode);
    const detailLabelBudget = Math.max(4, Math.min(32, Math.floor(state.scale * 4)));
    const candidates = visibleNodes.filter((node) => {
      if (node === focusNode || node.searchMatch) return true;
      if (isSchemeOverviewNode(node)) return true;
      if (node.type === "provider" && state.activeField && node.field === nodeById.get(state.activeField).field) return true;
      if (node.type === "article") return focusSet.has(node.id) && state.scale > 1.1;
      return node.type === "item" && focusSet.has(node.id) && state.scale > 0.7;
    }).sort((a, b) => {
      const priority = (node) => {
        if (node === focusNode) return 130;
        if (node.searchMatch) return 120;
        if (pathSet.has(node.id)) return 110 - node.level;
        if (isSchemeOverviewNode(node)) return 100 - node.level;
        if (focusSet.has(node.id)) {
          const distance = focusNode ? Math.hypot(node.x - focusNode.x, node.y - focusNode.y) : 0;
          return 80 - Math.min(24, distance / 18);
        }
        return node.type === "run" ? 70 : node.type === "group" ? 65 : node.type === "subgroup" ? 62 : node.type === "provider" ? 60 : 30;
      };
      return priority(b) - priority(a);
    });
    const occupied = [];
    let detailLabelsShown = 0;

    candidates.forEach((node) => {
      const detailCandidate = (node.type === "article" || node.type === "item") && !pathSet.has(node.id) && node !== focusNode;
      if (detailCandidate && detailLabelsShown >= detailLabelBudget) return;
      const point = worldToScreen(node.x, node.y);
      const radius = Math.max(2, node.size * Math.pow(state.scale, 0.68));
      const fontSize = node.type === "run" ? 12.5 : node.type === "group" ? 10.5 : node.type === "subgroup" ? 9.8 : node.type === "field" ? 9.2 : node.type === "article" ? 8.8 : node.type === "item" ? 7.8 : 8.3;
      const weight = HEADING_TYPES.has(node.type) ? 680 : 560;
      context.font = `${weight} ${fontSize}px Inter, ui-sans-serif, sans-serif`;
      const maxWidth = node.type === "run" ? 320 : node.type === "group" ? 190 : node.type === "subgroup" ? 168 : node.type === "field" ? 148 : node.type === "item" ? 240 : 210;
      const text = truncateLabel(graphDisplayLabel(node), maxWidth);
      const textWidth = context.measureText(text).width;
      const dimmed = focusSet.size > 0 && !focusSet.has(node.id);

      if (node.type === "field" && !state.search.trim()) {
        const angle = node.layoutAngle || Math.atan2(node.y, node.x);
        const flipped = Math.cos(angle) < 0;
        const rotation = flipped ? angle + Math.PI : angle;
        const localX = flipped ? -(radius + 7) : radius + 7;
        context.save();
        context.globalAlpha = dimmed ? 0.48 : 1;
        context.translate(point.x, point.y);
        context.rotate(rotation);
        context.textBaseline = "middle";
        context.textAlign = flipped ? "right" : "left";
        context.fillStyle = palette.labelBackground;
        context.fillRect(
          flipped ? localX - textWidth - 4 : localX - 4,
          -fontSize,
          textWidth + 8,
          fontSize * 2,
        );
        context.fillStyle = palette.labelMuted;
        context.fillText(text, localX, 0);
        context.restore();
        return;
      }

      let textX;
      let textY;
      let align;
      if (node.type === "run") {
        textX = point.x;
        textY = point.y + radius + 15;
        align = "center";
      } else {
        const direction = node.x >= 0 ? 1 : -1;
        textX = point.x + direction * (radius + 7);
        textY = point.y;
        align = direction > 0 ? "left" : "right";
      }
      const left = align === "left" ? textX - 4 : align === "right" ? textX - textWidth - 4 : textX - textWidth / 2 - 4;
      const box = { left, right: left + textWidth + 8, top: textY - fontSize, bottom: textY + fontSize };
      const essential = node === focusNode || node.searchMatch || pathSet.has(node.id) || isSchemeOverviewNode(node);
      if (!essential && occupied.some((other) => intersects(box, other))) return;
      occupied.push(box);
      if (detailCandidate) detailLabelsShown += 1;
      context.fillStyle = palette.labelBackground;
      context.fillRect(box.left, box.top, box.right - box.left, box.bottom - box.top);
      context.fillStyle = dimmed ? palette.labelMuted : node.searchMatch || HEADING_TYPES.has(node.type) ? palette.label : palette.labelMuted;
      context.textBaseline = "middle";
      context.textAlign = align;
      context.fillText(text, textX, textY);
    });
  }

  function draw() {
    const palette = canvasPalette();
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = palette.background;
    context.fillRect(0, 0, width, height);
    const focusSet = connectedSet(state.selectedNode || state.hoveredNode);
    const hasFocus = focusSet.size > 0;

    context.lineCap = "round";
    visibleEdges.forEach((edge) => {
      const source = worldToScreen(edge.sourceNode.x, edge.sourceNode.y);
      const target = worldToScreen(edge.targetNode.x, edge.targetNode.y);
      const focused = focusSet.has(edge.source) && focusSet.has(edge.target);
      const baseAlpha = edge.kind === "contains_group" ? 0.28 : edge.kind === "contains_field" ? 0.2 : edge.kind === "provider_branch" ? 0.18 : edge.kind === "article_coding" || edge.kind === "single_provider_article" ? 0.16 : 0.12;
      context.beginPath();
      context.moveTo(source.x, source.y);
      context.lineTo(target.x, target.y);
      context.strokeStyle = focused ? rgba(edge.targetNode.color, 0.72) : `rgba(${palette.edge},${hasFocus ? baseAlpha * palette.dimmedEdgeFactor : baseAlpha})`;
      context.lineWidth = focused ? 1.55 : Math.max(0.45, state.scale * 0.9);
      context.stroke();
    });

    visibleNodesDrawOrder.forEach((node) => {
      const point = worldToScreen(node.x, node.y);
      const radius = Math.max(1.7, node.size * Math.pow(state.scale, 0.68));
      if (point.x < -radius - 50 || point.y < -radius - 50 || point.x > width + radius + 50 || point.y > height + radius + 50) return;
      const selected = state.selectedNode === node;
      const hovered = state.hoveredNode === node;
      const focused = focusSet.has(node.id);
      const dimmed = hasFocus && !focused;
      const alpha = dimmed ? 0.32 : isSchemeOverviewNode(node) ? 0.98 : 0.88;
      if (selected || hovered) {
        context.beginPath();
        context.arc(point.x, point.y, radius + 6, 0, Math.PI * 2);
        context.fillStyle = rgba(node.color, selected ? 0.21 : 0.13);
        context.fill();
      }
      context.beginPath();
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      context.fillStyle = rgba(node.color, alpha);
      context.shadowColor = rgba(node.color, selected ? 0.9 : HEADING_TYPES.has(node.type) ? 0.42 : 0.2);
      context.shadowBlur = selected ? 17 : HEADING_TYPES.has(node.type) ? 9 : 4;
      context.fill();
      context.shadowBlur = 0;
      if (node.type === "group" || node.type === "subgroup" || node.type === "run") {
        context.beginPath();
        context.arc(point.x, point.y, radius + 3, 0, Math.PI * 2);
        context.strokeStyle = rgba(node.color, dimmed ? 0.18 : 0.48);
        context.lineWidth = 1;
        context.stroke();
      }
      if (node.fx !== null) {
        context.beginPath();
        context.arc(point.x, point.y, radius + 3.5, 0, Math.PI * 2);
        context.strokeStyle = rgba(node.color, 0.78);
        context.lineWidth = 1;
        context.stroke();
      }
    });
    drawLabels(palette, focusSet);
  }

  function loop(now) {
    const delta = now - lastTime;
    lastTime = now;
    simulate(delta);
    if (visibleNodes.length < 3500 || now - lastDrawTime > 66 || state.draggingNode) {
      draw();
      lastDrawTime = now;
    }
    requestAnimationFrame(loop);
  }

  function pointerPosition(event) {
    const bounds = canvas.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  function hitNode(screenPoint) {
    let best = null;
    let bestDistance = Infinity;
    visibleNodes.forEach((node) => {
      const point = worldToScreen(node.x, node.y);
      const distance = Math.hypot(point.x - screenPoint.x, point.y - screenPoint.y);
      const radius = Math.max(6, node.size * Math.pow(state.scale, 0.68) + 5);
      if (distance <= radius && distance < bestDistance) {
        best = node;
        bestDistance = distance;
      }
    });
    return best;
  }

  function showTooltip(node, point) {
    if (!node) {
      tooltip.style.display = "none";
      return;
    }
    tooltip.textContent = "";
    const strong = document.createElement("strong");
    strong.textContent = node.label;
    const span = document.createElement("span");
    span.textContent = [node.field_group, node.article_id, node.provider].filter(Boolean).join(" · ");
    tooltip.append(strong, span);
    tooltip.style.display = "block";
    const left = Math.min(width - 325, point.x + 14);
    const top = Math.min(height - 80, point.y + 14);
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${Math.max(8, top)}px`;
  }

  function visibleDescendants(node) {
    const result = [];
    const stack = [...(children.get(node.id) || [])];
    while (stack.length) {
      const id = stack.pop();
      const descendant = nodeById.get(id);
      if (!descendant) continue;
      if (descendant.visible) result.push(descendant);
      (children.get(id) || []).forEach((childId) => stack.push(childId));
    }
    return result;
  }

  function captureDraggingAnchor() {
    const node = state.draggingNode;
    if (!node || state.draggingAnchorCaptured) return;
    node.manualOffsetX += node.x - node.targetX;
    node.manualOffsetY += node.y - node.targetY;
    node.targetX = node.x;
    node.targetY = node.y;
    state.draggingAnchorCaptured = true;
  }

  canvas.addEventListener("pointermove", (event) => {
    const point = pointerPosition(event);
    if (state.draggingNode) {
      const world = screenToWorld(point.x, point.y);
      const previous = state.draggingLastWorld || { x: state.draggingNode.x, y: state.draggingNode.y };
      const dx = world.x - previous.x;
      const dy = world.y - previous.y;
      captureDraggingAnchor();
      state.draggingNode.targetX += dx;
      state.draggingNode.targetY += dy;
      state.draggingNode.x += dx;
      state.draggingNode.y += dy;
      state.draggingNode.manualOffsetX += dx;
      state.draggingNode.manualOffsetY += dy;
      state.draggingNode.fx = state.draggingNode.x;
      state.draggingNode.fy = state.draggingNode.y;
      state.draggingCluster.forEach((member) => {
        member.x += dx;
        member.y += dy;
        member.targetX += dx;
        member.targetY += dy;
        member.vx = 0;
        member.vy = 0;
        if (member.fx !== null) {
          member.fx += dx;
          member.fy += dy;
        }
      });
      state.draggingLastWorld = world;
      state.settledFrames = 0;
      return;
    }
    if (state.panning) {
      state.offsetX += point.x - state.pointerStart.x;
      state.offsetY += point.y - state.pointerStart.y;
      state.pointerStart = point;
      return;
    }
    const node = hitNode(point);
    state.hoveredNode = node;
    showTooltip(node, point);
    canvas.style.cursor = node ? "pointer" : "grab";
  });

  canvas.addEventListener("pointerdown", (event) => {
    const point = pointerPosition(event);
    const node = hitNode(point);
    state.pointerStart = point;
    if (node) {
      state.draggingNode = node;
      state.draggingCluster = visibleDescendants(node);
      state.draggingLastWorld = screenToWorld(point.x, point.y);
      state.draggingWasPinned = node.fx !== null;
      state.draggingAnchorCaptured = false;
      node.fx = node.x;
      node.fy = node.y;
    } else {
      state.panning = true;
    }
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointerup", (event) => {
    const point = pointerPosition(event);
    const node = hitNode(point);
    const moved = state.pointerStart && Math.hypot(point.x - state.pointerStart.x, point.y - state.pointerStart.y) > 5;
    if (state.draggingNode && !moved) {
      if (!state.draggingWasPinned) {
        state.draggingNode.fx = null;
        state.draggingNode.fy = null;
      }
      selectNode(state.draggingNode, point);
    }
    else if (!moved && node) selectNode(node, point);
    state.draggingNode = null;
    state.draggingCluster = [];
    state.draggingLastWorld = null;
    state.draggingWasPinned = false;
    state.draggingAnchorCaptured = false;
    state.panning = false;
    state.pointerStart = null;
    canvas.classList.remove("dragging");
    try { canvas.releasePointerCapture(event.pointerId); } catch (error) { /* Pointer is already released. */ }
  });

  canvas.addEventListener("dblclick", (event) => {
    const node = hitNode(pointerPosition(event));
    if (node && ["group", "field", "article", "provider"].includes(node.type)) toggleExpansion(node);
  });

  canvas.addEventListener("pointerleave", () => {
    if (!state.draggingNode && !state.panning) {
      state.hoveredNode = null;
      tooltip.style.display = "none";
    }
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const point = pointerPosition(event);
    const before = screenToWorld(point.x, point.y);
    const boundedDelta = Math.max(-80, Math.min(80, event.deltaY));
    const factor = Math.exp(-boundedDelta * 0.0016);
    state.scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, state.scale * factor));
    const after = worldToScreen(before.x, before.y);
    state.offsetX += point.x - after.x;
    state.offsetY += point.y - after.y;
    zoomReadout.textContent = `${Math.round(state.scale * 100)}%`;
  }, { passive: false });

  function humanize(value) {
    return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function displayValue(value) {
    if (value === null || value === undefined || value === "") return "Not recorded";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    const text = String(value);
    return /^[a-z0-9_ ]+$/.test(text) && text.includes("_") ? text.replaceAll("_", " ") : text;
  }

  function isPrimitive(value) {
    return value === null || value === undefined || typeof value !== "object";
  }

  function appendDetailValue(host, label, value, options) {
    const settings = options || {};
    const section = document.createElement("section");
    section.className = "detail-section";
    if (label) {
      const heading = document.createElement("h3");
      heading.className = "detail-section-title";
      heading.textContent = humanize(label);
      section.appendChild(heading);
    }

    if (Array.isArray(value)) {
      if (!value.length) {
        const empty = document.createElement("p");
        empty.className = "detail-value empty";
        empty.textContent = "No entries recorded";
        section.appendChild(empty);
      } else if (value.every(isPrimitive)) {
        const list = document.createElement("div");
        list.className = "detail-chip-list";
        value.slice(0, settings.compact ? 8 : value.length).forEach((item) => {
          const chip = document.createElement("span");
          chip.className = "detail-chip";
          chip.textContent = displayValue(item);
          list.appendChild(chip);
        });
        section.appendChild(list);
      } else {
        const limit = settings.compact ? 1 : value.length;
        value.slice(0, limit).forEach((item, index) => {
          const entryLabel = document.createElement("p");
          entryLabel.className = "detail-array-label";
          entryLabel.textContent = `Entry ${index + 1}`;
          section.appendChild(entryLabel);
          const object = document.createElement("div");
          object.className = "detail-object";
          Object.entries(item || {}).forEach(([key, nested]) => appendDetailValue(object, key, nested, settings));
          section.appendChild(object);
        });
        if (settings.compact && value.length > limit) {
          const remaining = document.createElement("p");
          remaining.className = "detail-value empty";
          remaining.textContent = `${value.length - limit} more entries in the full inspector`;
          section.appendChild(remaining);
        }
      }
    } else if (value && typeof value === "object") {
      const object = document.createElement("div");
      object.className = "detail-object";
      const entries = Object.entries(value);
      entries.slice(0, settings.compact ? 4 : entries.length).forEach(([key, nested]) => appendDetailValue(object, key, nested, settings));
      section.appendChild(object);
    } else {
      const paragraph = document.createElement("p");
      const quote = /quote|verbatim|evidence excerpt/i.test(label || "");
      paragraph.className = `detail-value${quote ? " quote" : ""}${value === null || value === undefined || value === "" ? " empty" : ""}`;
      paragraph.textContent = displayValue(value);
      section.appendChild(paragraph);
    }
    host.appendChild(section);
  }

  function renderDetails(detail, host, compact) {
    host.textContent = "";
    const source = detail && typeof detail === "object" && !Array.isArray(detail) ? detail : { Value: detail };
    const entries = Object.entries(source);
    const limit = compact ? 4 : entries.length;
    entries.slice(0, limit).forEach(([key, value]) => appendDetailValue(host, key, value, { compact }));
  }

  function pathFor(node) {
    const parts = [];
    let cursor = node;
    while (cursor) {
      parts.unshift(cursor.type === "run" ? "Run" : cursor.label);
      cursor = parents.has(cursor.id) ? nodeById.get(parents.get(cursor.id)) : null;
    }
    return parts.join(" / ");
  }

  function displayParent(node) {
    let parent = parents.has(node.id) ? nodeById.get(parents.get(node.id)) : null;
    while (parent && !parent.visible) {
      parent = parents.has(parent.id) ? nodeById.get(parents.get(parent.id)) : null;
    }
    return parent;
  }

  function filteredNeighbors(node) {
    const candidates = [];
    const parent = displayParent(node);
    if (parent) candidates.push(parent);
    (children.get(node.id) || []).forEach((id) => {
      const child = nodeById.get(id);
      if (child && child.visible) candidates.push(child);
    });
    if (node.type === "field" && state.selectedProviders.size === 1) {
      (children.get(node.id) || []).forEach((providerId) => {
        const provider = nodeById.get(providerId);
        if (!provider || !state.selectedProviders.has(provider.provider)) return;
        (children.get(provider.id) || []).forEach((articleId) => {
          const article = nodeById.get(articleId);
          if (article && article.visible) candidates.push(article);
        });
      });
    }
    return Array.from(new Map(candidates.map((item) => [item.id, item])).values());
  }

  function expansionAction(node) {
    if (state.showAll) return null;
    if ((node.type === "group" || node.type === "subgroup") && state.activeField) {
      return { label: "Back to scheme overview", expanded: true };
    }
    if (node.type === "field") {
      const expanded = state.activeField === node.id;
      const providerNodes = (children.get(node.id) || [])
        .map((id) => nodeById.get(id))
        .filter((item) => item && passesFilters(item));
      const paperCount = providerNodes.reduce((total, provider) => total + (children.get(provider.id) || [])
        .map((id) => nodeById.get(id))
        .filter((article) => article && passesFilters(article)).length, 0);
      const label = providerNodes.length === 1
        ? `Explore ${paperCount} papers`
        : `Explore ${providerNodes.length} providers · ${paperCount} papers`;
      return { label: expanded ? "Collapse field" : label, expanded };
    }
    if (node.type === "article") {
      const expanded = state.activeArticle === node.id;
      const count = (children.get(node.id) || []).map((id) => nodeById.get(id)).filter((item) => item && passesFilters(item)).length;
      if (!count) return null;
      return { label: expanded ? "Hide extracted items" : `Show ${count} extracted items`, expanded };
    }
    return null;
  }

  function configureActionButton(button, node) {
    const action = expansionAction(node);
    button.hidden = !action;
    if (action) {
      button.textContent = action.label;
      button.setAttribute("aria-pressed", String(action.expanded));
    }
  }

  function configureBackButton(node) {
    const button = document.getElementById("backNode");
    const parent = displayParent(node);
    button.hidden = !parent;
    if (!parent) return;
    const parentLabel = parent.type === "run" ? "scheme root" : parent.label;
    button.textContent = `Back to ${parentLabel.length > 34 ? `${parentLabel.slice(0, 32)}...` : parentLabel}`;
  }

  function showQuickCard(node, screenPoint) {
    if (!state.popupEnabled) {
      quickCard.hidden = true;
      return;
    }
    document.getElementById("quickCardType").textContent = node.type;
    document.getElementById("quickCardType").style.borderColor = rgba(node.color, 0.58);
    document.getElementById("quickCardTitle").textContent = node.label;
    document.getElementById("quickCardMeta").textContent = [node.field_group, node.article_id, node.provider].filter(Boolean).join(" · ") || pathFor(node);
    renderDetails(node.detail || { Value: node.value }, document.getElementById("quickCardBody"), true);
    configureActionButton(document.getElementById("quickCardExpand"), node);
    quickCard.hidden = false;
    quickCard.dataset.nodeId = node.id;
    const cardWidth = 360;
    const cardHeight = Math.min(460, quickCard.scrollHeight || 320);
    let left = screenPoint.x + 18;
    let top = screenPoint.y + 18;
    if (left + cardWidth > width - 12) left = screenPoint.x - cardWidth - 18;
    if (top + cardHeight > height - 12) top = height - cardHeight - 12;
    quickCard.style.left = `${Math.max(12, left)}px`;
    quickCard.style.top = `${Math.max(12, top)}px`;
  }

  function populateInspector(node) {
    document.getElementById("detailEmpty").hidden = true;
    detailContent.hidden = false;
    detailContent.scrollTop = 0;
    document.getElementById("detailType").textContent = node.type;
    document.getElementById("detailType").style.borderColor = rgba(node.color, 0.58);
    document.getElementById("detailTitle").textContent = node.label;
    document.getElementById("detailPath").textContent = pathFor(node);
    configureBackButton(node);
    configureActionButton(document.getElementById("expandNode"), node);
    renderDetails(node.detail || { Value: node.value }, document.getElementById("detailFields"), false);

    const neighborHost = document.getElementById("neighborList");
    neighborHost.textContent = "";
    const neighbors = filteredNeighbors(node);
    document.getElementById("neighborSummary").textContent = `${neighbors.length.toLocaleString()} immediate connections`;
    neighbors.slice(0, 60).forEach((neighbor) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "neighbor-button";
      button.textContent = `${humanize(neighbor.type)} · ${neighbor.label}`;
      button.addEventListener("click", () => revealAndSelect(neighbor));
      neighborHost.appendChild(button);
    });
    if (neighbors.length > 60) {
      const note = document.createElement("p");
      note.className = "detail-value empty";
      note.textContent = `${neighbors.length - 60} additional connections are available through the graph filters.`;
      neighborHost.appendChild(note);
    }
  }

  function selectNode(node, screenPoint) {
    const sameNode = state.selectedNode === node;
    state.selectedNode = node;
    populateInspector(node);
    if (sameNode && !quickCard.hidden) {
      quickCard.hidden = true;
      return;
    }
    showQuickCard(node, screenPoint || worldToScreen(node.x, node.y));
  }

  function revealAndSelect(node) {
    state.search = "";
    searchInput.value = "";
    if (node.type === "provider") {
      const field = nodeById.get(parents.get(node.id));
      state.activeField = field.id;
      state.activeProvider = node.id;
      state.activeArticle = null;
    } else if (node.type === "article") {
      const provider = nodeById.get(parents.get(node.id));
      const field = nodeById.get(parents.get(provider.id));
      state.activeField = field.id;
      state.activeProvider = provider.id;
      state.activeArticle = null;
    } else if (node.type === "item") {
      const article = nodeById.get(parents.get(node.id));
      const provider = nodeById.get(parents.get(article.id));
      const field = nodeById.get(parents.get(provider.id));
      state.activeField = field.id;
      state.activeProvider = provider.id;
      state.activeArticle = article.id;
    }
    initializePositions();
    updateVisibility();
    selectNode(node, worldToScreen(node.x, node.y));
    requestAnimationFrame(fitVisible);
  }

  function clearDetail() {
    state.selectedNode = null;
    state.hoveredNode = null;
    detailContent.hidden = true;
    document.getElementById("detailEmpty").hidden = false;
    quickCard.hidden = true;
    tooltip.style.display = "none";
  }

  function cameraNodes() {
    if (!state.activeField || state.showAll || state.search.trim()) return visibleNodes;
    const field = nodeById.get(state.activeField);
    if (!field) return visibleNodes;
    const branch = [field];

    if (state.activeArticle) {
      const article = nodeById.get(state.activeArticle);
      if (!article) return branch;
      const provider = nodeById.get(parents.get(article.id));
      if (provider && provider.visible) branch.push(provider);
      if (provider) {
        (children.get(provider.id) || []).forEach((id) => {
          const sibling = nodeById.get(id);
          if (sibling && sibling.visible) branch.push(sibling);
        });
      }
      (children.get(article.id) || []).forEach((id) => {
        const item = nodeById.get(id);
        if (item && item.visible) branch.push(item);
      });
      return branch;
    }

    if (state.activeProvider) {
      const provider = nodeById.get(state.activeProvider);
      if (!provider) return branch;
      if (provider.visible) branch.push(provider);
      (children.get(provider.id) || []).forEach((id) => {
        const article = nodeById.get(id);
        if (article && article.visible) branch.push(article);
      });
      return branch;
    }

    visibleNodes.forEach((node) => {
      if ((node.type === "provider" || node.type === "article") && node.field === field.field) branch.push(node);
    });
    return branch;
  }

  function fitVisible() {
    const framedNodes = cameraNodes();
    if (!framedNodes.length) return;
    const xs = framedNodes.map((node) => node.x);
    const ys = framedNodes.map((node) => node.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const overview = !state.activeField && !state.search.trim() && !state.showAll;
    const horizontalInset = overview ? 330 : 190;
    const verticalInset = overview ? 210 : 170;
    const graphWidth = Math.max(140, maxX - minX);
    const graphHeight = Math.max(140, maxY - minY);
    const availableWidth = Math.max(180, width - horizontalInset);
    const availableHeight = Math.max(180, height - verticalInset);
    const autoScaleLimit = state.activeArticle ? 2.8 : state.activeProvider ? 2 : state.activeField ? 1.5 : 2.1;
    state.scale = Math.max(MIN_SCALE, Math.min(autoScaleLimit, Math.min(availableWidth / graphWidth, availableHeight / graphHeight) * 0.96));
    state.offsetX = -(minX + maxX) / 2 * state.scale;
    state.offsetY = -(minY + maxY) / 2 * state.scale;
    zoomReadout.textContent = `${Math.round(state.scale * 100)}%`;
  }

  function toggleExpansion(node) {
    if (state.showAll) return;
    const collapsingField = node.type === "field" && state.activeField === node.id;
    if (node.type === "group" || node.type === "subgroup") {
      state.activeField = null;
      state.activeArticle = null;
      state.activeProvider = null;
    } else if (node.type === "field") {
      state.activeField = collapsingField ? null : node.id;
      state.activeArticle = null;
      state.activeProvider = null;
    } else if (node.type === "provider") {
      const field = nodeById.get(parents.get(node.id));
      state.activeField = field.id;
      state.activeProvider = node.id;
      state.activeArticle = null;
    } else if (node.type === "article" && (children.get(node.id) || []).length) {
      const provider = nodeById.get(parents.get(node.id));
      const field = nodeById.get(parents.get(provider.id));
      state.activeField = field.id;
      state.activeProvider = provider.id;
      state.activeArticle = state.activeArticle === node.id ? null : node.id;
    } else {
      return;
    }
    state.search = "";
    searchInput.value = "";
    initializePositions();
    updateVisibility();
    if (collapsingField || node.type === "group" || node.type === "subgroup") {
      clearDetail();
    } else {
      state.selectedNode = node;
      populateInspector(node);
      showQuickCard(node, worldToScreen(node.x, node.y));
    }
    requestAnimationFrame(fitVisible);
  }

  function navigateBack() {
    const node = state.selectedNode;
    if (!node) return;
    const parent = displayParent(node);
    if (!parent) return;
    if (node.type === "item") {
      state.activeArticle = null;
    } else if (node.type === "provider") {
      state.activeProvider = null;
    } else if (node.type === "article") {
      state.activeArticle = null;
      state.activeProvider = parent.type === "provider" ? parent.id : null;
    } else if (node.type === "field" || node.type === "group" || node.type === "subgroup") {
      state.activeField = null;
      state.activeArticle = null;
      state.activeProvider = null;
    }
    initializePositions();
    updateVisibility();
    selectNode(parent, worldToScreen(parent.x, parent.y));
    requestAnimationFrame(fitVisible);
  }

  function resetGraph() {
    state.activeField = null;
    state.activeArticle = null;
    state.activeProvider = null;
    state.showAll = false;
    state.search = "";
    searchInput.value = "";
    document.querySelectorAll("input[data-filter-group]").forEach((input) => { input.checked = true; });
    state.selectedArticles = new Set(data.filters.articles.map((item) => item.id));
    state.selectedProviders = new Set(data.filters.providers.map((item) => item.id));
    state.selectedFields = new Set(allFieldIds);
    document.getElementById("showAll").textContent = "Show all";
    document.getElementById("showAll").setAttribute("aria-pressed", "false");
    nodes.forEach((node) => {
      node.fx = null;
      node.fy = null;
      node.vx = 0;
      node.vy = 0;
      node.manualOffsetX = 0;
      node.manualOffsetY = 0;
    });
    initializePositions();
    updateVisibility();
    clearDetail();
    requestAnimationFrame(fitVisible);
  }

  function toggleShowAll() {
    state.showAll = !state.showAll;
    state.activeField = null;
    state.activeArticle = null;
    state.activeProvider = null;
    state.search = "";
    searchInput.value = "";
    if (state.showAll) {
      document.querySelectorAll("input[data-filter-group]").forEach((input) => { input.checked = true; });
      state.selectedArticles = new Set(data.filters.articles.map((item) => item.id));
      state.selectedProviders = new Set(data.filters.providers.map((item) => item.id));
      state.selectedFields = new Set(allFieldIds);
    }
    const button = document.getElementById("showAll");
    button.textContent = state.showAll ? "Close all" : "Show all";
    button.setAttribute("aria-pressed", String(state.showAll));
    nodes.forEach((node) => {
      node.fx = null;
      node.fy = null;
      node.manualOffsetX = 0;
      node.manualOffsetY = 0;
    });
    initializePositions();
    updateVisibility();
    clearDetail();
    requestAnimationFrame(fitVisible);
  }

  function setPopupEnabled(enabled) {
    state.popupEnabled = enabled;
    const button = document.getElementById("popupToggle");
    button.textContent = enabled ? "Pop-up on" : "Pop-up off";
    button.setAttribute("aria-pressed", String(enabled));
    if (!enabled) quickCard.hidden = true;
    try { window.localStorage.setItem("bpsGraphPopup", enabled ? "on" : "off"); } catch (error) { /* Local storage can be unavailable for local files. */ }
  }

  function initialPopupSetting() {
    try { return window.localStorage.getItem("bpsGraphPopup") !== "off"; } catch (error) { return true; }
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const light = theme === "light";
    const button = document.getElementById("themeToggle");
    button.textContent = light ? "Dark" : "Light";
    button.setAttribute("aria-pressed", String(light));
    try { window.localStorage.setItem("bpsGraphTheme", theme); } catch (error) { /* Local storage can be unavailable for local files. */ }
  }

  function initialTheme() {
    try {
      const stored = window.localStorage.getItem("bpsGraphTheme");
      if (stored === "light" || stored === "dark") return stored;
    } catch (error) { /* Local storage can be unavailable for local files. */ }
    return "dark";
  }

  searchInput.addEventListener("input", () => {
    state.search = searchInput.value;
    updateVisibility();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== searchInput) {
      event.preventDefault();
      searchInput.focus();
    }
    if (event.key === "Escape") clearDetail();
  });
  document.getElementById("fitGraph").addEventListener("click", fitVisible);
  document.getElementById("resetGraph").addEventListener("click", resetGraph);
  document.getElementById("showAll").addEventListener("click", toggleShowAll);
  document.getElementById("popupToggle").addEventListener("click", () => setPopupEnabled(!state.popupEnabled));
  document.getElementById("toggleLabels").addEventListener("click", (event) => {
    state.showLabels = !state.showLabels;
    event.currentTarget.setAttribute("aria-pressed", String(state.showLabels));
  });
  document.getElementById("themeToggle").addEventListener("click", () => {
    setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
  });
  document.getElementById("closeDetail").addEventListener("click", clearDetail);
  document.getElementById("quickCardClose").addEventListener("click", () => { quickCard.hidden = true; });
  document.getElementById("quickCardOpen").addEventListener("click", () => {
    if (state.selectedNode) {
      populateInspector(state.selectedNode);
      detailContent.focus({ preventScroll: true });
    }
  });
  document.getElementById("quickCardExpand").addEventListener("click", () => {
    if (state.selectedNode) toggleExpansion(state.selectedNode);
  });
  document.getElementById("expandNode").addEventListener("click", () => {
    if (state.selectedNode) toggleExpansion(state.selectedNode);
  });
  document.getElementById("backNode").addEventListener("click", navigateBack);
  document.getElementById("focusNode").addEventListener("click", fitVisible);
  document.getElementById("releaseNode").addEventListener("click", () => {
    if (state.selectedNode) {
      state.selectedNode.fx = null;
      state.selectedNode.fy = null;
      state.settledFrames = 0;
    }
  });
  const quickCardHandle = document.getElementById("quickCardHandle");
  quickCardHandle.addEventListener("pointerdown", (event) => {
    if (event.target.closest("button")) return;
    state.quickCardDragging = true;
    state.quickCardDragStart = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      left: parseFloat(quickCard.style.left) || quickCard.offsetLeft,
      top: parseFloat(quickCard.style.top) || quickCard.offsetTop,
    };
    quickCard.classList.add("dragging");
    quickCardHandle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  quickCardHandle.addEventListener("pointermove", (event) => {
    if (!state.quickCardDragging || !state.quickCardDragStart) return;
    const nextLeft = state.quickCardDragStart.left + event.clientX - state.quickCardDragStart.pointerX;
    const nextTop = state.quickCardDragStart.top + event.clientY - state.quickCardDragStart.pointerY;
    quickCard.style.left = `${Math.max(8, Math.min(width - quickCard.offsetWidth - 8, nextLeft))}px`;
    quickCard.style.top = `${Math.max(8, Math.min(height - quickCard.offsetHeight - 8, nextTop))}px`;
  });
  function stopQuickCardDrag(event) {
    if (!state.quickCardDragging) return;
    state.quickCardDragging = false;
    state.quickCardDragStart = null;
    quickCard.classList.remove("dragging");
    try { quickCardHandle.releasePointerCapture(event.pointerId); } catch (error) { /* Pointer is already released. */ }
  }
  quickCardHandle.addEventListener("pointerup", stopQuickCardDrag);
  quickCardHandle.addEventListener("pointercancel", stopQuickCardDrag);
  window.addEventListener("resize", resizeCanvas);

  setTheme(initialTheme());
  setPopupEnabled(initialPopupSetting());
  buildHeadline();
  buildFilters();
  resizeCanvas();
  initializePositions();
  updateVisibility();
  requestAnimationFrame(() => {
    fitVisible();
    requestAnimationFrame(loop);
  });
}());
