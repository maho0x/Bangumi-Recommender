<script setup>
import { ref, computed } from 'vue'

const username = ref('')
const subjectType = ref(2)
const includeNsfw = ref(false)
const loading = ref(false)
const error = ref('')
const result = ref(null)
const userProfile = ref(null)

const subjectTypes = [
  { value: 2, label: '动画' },
  { value: 1, label: '书籍' },
  { value: 3, label: '音乐' },
  { value: 4, label: '游戏' },
  { value: 6, label: '三次元' },
]

const sortBy = ref('score')
const recommendations = computed(() => {
  if (!result.value?.recommendations) return []
  const recs = [...result.value.recommendations]
  if (sortBy.value === 'bangumi_score') {
    recs.sort((a, b) => (b.bangumi_score || 0) - (a.bangumi_score || 0))
  }
  return recs
})

async function fetchRecommendations() {
  if (!username.value.trim()) return
  loading.value = true
  error.value = ''
  result.value = null

  try {
    // Fetch profile in parallel
    fetch(`/api/user/${encodeURIComponent(username.value)}/profile`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { userProfile.value = data })
      .catch(() => {})

    const params = new URLSearchParams({
      username: username.value.trim(),
      subject_type: subjectType.value,
      limit: 20,
      nsfw: includeNsfw.value,
    })
    const resp = await fetch(`/api/recommend?${params}`)
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.detail || `HTTP ${resp.status}`)
    }
    result.value = await resp.json()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

function scoreColor(score) {
  if (score >= 0.8) return 'text-green-600'
  if (score >= 0.6) return 'text-blue-600'
  if (score >= 0.4) return 'text-yellow-600'
  return 'text-gray-500'
}

function bangumiUrl(id) {
  return `https://bgm.tv/subject/${id}`
}
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 shadow-sm">
      <div class="max-w-6xl mx-auto px-4 py-6">
        <h1 class="text-2xl font-bold text-gray-900">Bangumi 推荐系统</h1>
        <p class="text-gray-500 mt-1">基于协同过滤和内容分析的个性化条目推荐</p>
      </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 py-8">
      <!-- Search Form -->
      <div class="bg-white rounded-lg shadow p-6 mb-8">
        <div class="flex flex-wrap gap-4 items-end">
          <div class="flex-1 min-w-[200px]">
            <label class="block text-sm font-medium text-gray-700 mb-1">Bangumi 用户名</label>
            <input
              v-model="username"
              type="text"
              placeholder="输入你的 bgm.tv 用户名"
              class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              @keyup.enter="fetchRecommendations"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">条目类型</label>
            <select
              v-model="subjectType"
              class="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
            >
              <option v-for="t in subjectTypes" :key="t.value" :value="t.value">
                {{ t.label }}
              </option>
            </select>
          </div>
          <div class="flex items-center gap-2">
            <input id="nsfw" v-model="includeNsfw" type="checkbox" class="rounded" />
            <label for="nsfw" class="text-sm text-gray-600">包含 NSFW</label>
          </div>
          <button
            @click="fetchRecommendations"
            :disabled="loading || !username.trim()"
            class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <span v-if="loading">加载中...</span>
            <span v-else>获取推荐</span>
          </button>
        </div>
      </div>

      <!-- Error -->
      <div v-if="error" class="bg-red-50 border border-red-200 rounded-lg p-4 mb-8 text-red-700">
        {{ error }}
      </div>

      <!-- Loading -->
      <div v-if="loading" class="text-center py-20">
        <div class="inline-block w-8 h-8 border-4 border-blue-300 border-t-blue-600 rounded-full animate-spin"></div>
        <p class="mt-4 text-gray-500">正在获取收藏数据并计算推荐...</p>
        <p class="text-sm text-gray-400 mt-1">首次获取可能需要 3-10 秒</p>
      </div>

      <!-- User Profile -->
      <div v-if="userProfile && result" class="bg-white rounded-lg shadow p-6 mb-6">
        <h2 class="text-lg font-semibold text-gray-800 mb-3">
          {{ userProfile.username }} 的收藏概览
        </h2>
        <div class="flex flex-wrap gap-4 text-sm">
          <span v-if="userProfile.anime_count" class="px-3 py-1 bg-blue-50 text-blue-700 rounded-full">
            动画 {{ userProfile.anime_count }}
          </span>
          <span v-if="userProfile.book_count" class="px-3 py-1 bg-green-50 text-green-700 rounded-full">
            书籍 {{ userProfile.book_count }}
          </span>
          <span v-if="userProfile.music_count" class="px-3 py-1 bg-purple-50 text-purple-700 rounded-full">
            音乐 {{ userProfile.music_count }}
          </span>
          <span v-if="userProfile.game_count" class="px-3 py-1 bg-orange-50 text-orange-700 rounded-full">
            游戏 {{ userProfile.game_count }}
          </span>
          <span v-if="userProfile.real_count" class="px-3 py-1 bg-pink-50 text-pink-700 rounded-full">
            三次元 {{ userProfile.real_count }}
          </span>
          <span v-if="userProfile.avg_rating" class="px-3 py-1 bg-gray-100 text-gray-700 rounded-full">
            平均评分 {{ userProfile.avg_rating.toFixed(1) }}
          </span>
        </div>
      </div>

      <!-- Results -->
      <div v-if="result && !loading">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-gray-800">
            推荐结果
            <span class="text-sm font-normal text-gray-500">
              (共 {{ recommendations.length }} 个，基于 {{ result.total_collections }} 条收藏)
            </span>
          </h2>
          <div class="flex items-center gap-2 text-sm">
            <span class="text-gray-500">排序:</span>
            <select v-model="sortBy" class="px-2 py-1 border border-gray-300 rounded text-sm">
              <option value="score">推荐度</option>
              <option value="bangumi_score">社区评分</option>
            </select>
            <span v-if="result.cf_available" class="ml-2 px-2 py-1 bg-green-50 text-green-700 rounded text-xs">
              CF 可用
            </span>
          </div>
        </div>

        <!-- Recommendation Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          <div
            v-for="(rec, idx) in recommendations"
            :key="rec.subject_id"
            class="bg-white rounded-lg shadow hover:shadow-md transition-shadow overflow-hidden"
          >
            <!-- Cover Image -->
            <a :href="bangumiUrl(rec.subject_id)" target="_blank" class="block">
              <div class="h-48 bg-gray-100 flex items-center justify-center overflow-hidden">
                <img
                  :src="rec.image_url"
                  :alt="rec.name_cn || rec.name"
                  class="w-full h-full object-cover"
                  loading="lazy"
                  @error="$event.target.style.display='none'"
                />
              </div>
            </a>

            <div class="p-4">
              <!-- Title -->
              <a :href="bangumiUrl(rec.subject_id)" target="_blank" class="block mb-2 hover:text-blue-600">
                <h3 class="font-semibold text-gray-900 text-sm line-clamp-1">
                  {{ rec.name_cn || rec.name }}
                </h3>
                <p v-if="rec.name_cn && rec.name && rec.name_cn !== rec.name" class="text-xs text-gray-400 line-clamp-1">
                  {{ rec.name }}
                </p>
              </a>

              <!-- Scores -->
              <div class="flex items-center gap-3 mb-2 text-sm">
                <span :class="scoreColor(rec.score)">
                  推荐度 {{ (rec.score * 100).toFixed(0) }}%
                </span>
                <span v-if="rec.bangumi_score" class="text-gray-500">
                  ★ {{ rec.bangumi_score }}
                </span>
              </div>

              <!-- Tags -->
              <div v-if="rec.tags?.length" class="flex flex-wrap gap-1 mb-2">
                <span
                  v-for="tag in rec.tags.slice(0, 5)"
                  :key="tag"
                  class="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded"
                >
                  {{ tag }}
                </span>
              </div>

              <!-- Reason -->
              <p v-if="rec.reason" class="text-xs text-gray-500 line-clamp-2">
                {{ rec.reason }}
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>
