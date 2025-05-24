# 📝 LeetCode Problem Notes

## 📑 Table of Contents

### Arrays & Hashing

- [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate)
- [242. Valid Anagram](https://leetcode.com/problems/valid-anagram)
- [1. Two Sum](https://leetcode.com/problems/two-sum)
- [49. Group Anagrams](https://leetcode.com/problems/group-anagrams)
- [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements)

---

## 🧩 [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate)

### 📋 My Approach

Sort the array and check if any two adjacent elements are the same.

### 💻 My Solution ([Screenshot](./neetcode/ContainsDuplicate.png))

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums.sort()
        n = len(nums)
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                return True
        return False
```

### 📊 Complexity Analysis

- **Time Complexity:** O(n log n)
- **Space Complexity:** O(1)

### 🔄 Possible Improvements

- Use a hash set to detect duplicates. This is O(1) time per check/insert.

---

## 🧩 [242. Valid Anagram](https://leetcode.com/problems/valid-anagram)

### 📋 My Approach

Sort the strings and check if they are equal.

### 💻 My Solution

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s = sorted(s)
        t = sorted(t)
        return s == t
```

### 📊 Complexity Analysis

- **Time Complexity:** O(n log n)
- **Space Complexity:** O(1)

### 🔄 Possible Improvements

- Use a hash map to count the frequency of each character. This is O(n) time.

---

## 🧩 [1. Two Sum](https://leetcode.com/problems/two-sum)

### 📋 My Approach

Use a hash map to store the indices of the numbers.

### 💻 My Solution

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        nums = enumerate(nums)
        for i, num in nums:
            if target - num in hashmap:
                return [i, hashmap[target - num]]
            hashmap[num] = i
```

### 📊 Complexity Analysis

- **Time Complexity:** O(n)
- **Space Complexity:** O(n)

### 🔄 Possible Improvements

- This is optimal for this problem.

---

## 🧩 [49. Group Anagrams](https://leetcode.com/problems/group-anagrams)

### 📋 My Approach

Use a hash map to store the indices of the sorted strings.

### 💻 My Solution

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        map = {}
        result = []

        for s in strs:
            sorted_str = ''.join(sorted(s))
            if sorted_str in map:
                result[map[sorted_str]].append(s)
            else:
                map[sorted_str] = len(result)
                result.append([s])

        return result
```

### 📊 Complexity Analysis

- **Time Complexity:** O(n)
- **Space Complexity:** O(n)

### 🔄 Possible Improvements

- This is optimal for this problem.

---

## 🧩 [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements)

### 📋 My Approach

Use a hash map to keep track of the frequency of each number.

### 💻 My Solution

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        for num in nums:
            count[num] = 1 + count.get(num, 0)

        arr = []
        for num, index in count.items():
            arr.append([index, num])
        arr.sort()

        result = []
        while len(result) < k:
            result.append(arr.pop()[1])
        return result
```

### 📊 Complexity Analysis

- **Time Complexity:** O(n)
- **Space Complexity:** O(n)

### 🔄 Possible Improvements

- This is optimal for this problem.

---

## 📚 Key Takeaways & Lessons Learned for this section

- Hash-based lookups give O(1) average time per check/insert.
- Sorting is O(n log n) time.
- Use a hash map! (to store the indices of the sorted strings, to keep track of the frequency of each number, to store the indices of the numbers)
