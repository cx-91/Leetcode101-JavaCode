---


---

<h1 id="leetcode-101">Leetcode 101</h1>
<h2 id="第二章----最易懂的贪心算法">第二章 – 最易懂的贪心算法</h2>
<h4 id="assign-cookies">455. Assign Cookies</h4>
<pre><code>public int findContentChildren(int[] g, int[] s) {
    Arrays.sort(g);
    Arrays.sort(s);
    int j = 0, i = 0;
    while(j &lt; s.length &amp;&amp; i &lt; g.length){
        if(s[j] &gt;= g[i]) i++;
        j++;
    }
    return i;
}
</code></pre>
<h4 id="non-overlapping-intervals">435. Non-overlapping Intervals</h4>
<pre><code>public int eraseOverlapIntervals(int[][] intervals) {
    if(intervals.length == 0) return 0;
    
    Arrays.sort(intervals, (a, b) -&gt; a[1] - b[1]);
    for(int[] i : intervals){
        System.out.println(i[0] + " " + i[1]);
    }
    int removed = 0, prev = intervals[0][1];
    for(int i = 1; i &lt; intervals.length; i++){
        if(intervals[i][0] &lt; prev){
            removed++;
        }
        else{
            prev = intervals[i][1];
        }
    }
    return removed;
}
</code></pre>
<h2 id="第三章----玩转双指针">第三章 – 玩转双指针</h2>
<h4 id="two-sum-ii---input-array-is-sorted">167. Two Sum II - Input array is sorted</h4>
<pre><code>public int[] twoSum(int[] numbers, int target) {
    int l = 0, r = numbers.length - 1, sum;
    while(l &lt; r){
        sum = numbers[l] + numbers[r];
        if(sum == target) break;
        if(sum &lt; target) l++;
        else r--;
    }
    return new int[]{l + 1, r + 1};
}
</code></pre>
<h4 id="merge-sorted-array">88. Merge Sorted Array</h4>
<pre><code>public void merge(int[] nums1, int m, int[] nums2, int n) {
    int pos = m-- + n-- - 1;
    while(m &gt;= 0 &amp;&amp; n &gt;= 0){
        nums1[pos--] = nums1[m] &gt; nums2[n] ? nums1[m--] : nums2[n--];
    }
    while(n &gt;= 0){
        nums1[pos--] = nums2[n--];
    }
}
</code></pre>
<p>直观的写法：</p>
<pre><code>public void merge(int[] nums1, int m, int[] nums2, int n) {
    if(m == 0 &amp;&amp; n == 0) return;
    if(m != 0 &amp;&amp; n != 0){
        if(nums1[m - 1] &lt; nums2[n - 1]){
            nums1[m + n - 1] = nums2[n - 1];
            n--;
        }
        else{
            nums1[m + n - 1] = nums1[m - 1];
            m--;
        }
    }
    if(m != 0 &amp;&amp; n == 0){
        return;
    }
    if(m == 0 &amp;&amp; n != 0){
        nums1[m + n - 1] = nums2[n - 1];
        n--;
    }
    merge(nums1, m, nums2, n);
}
</code></pre>
<h4 id="linked-list-cycle-ii">142. Linked List Cycle II</h4>
<pre><code> public ListNode detectCycle(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;
  
    do{
        if(fast == null || fast.next == null) return null;
        fast = fast.next.next;
        slow = slow.next;
    } while(fast != slow);
    
    fast = head;
    while(fast != slow){
        slow = slow.next;
        fast = fast.next;
    }
    return fast;
}
</code></pre>
<h4 id="minimum-window-substring">76. Minimum Window Substring</h4>
<pre><code>  public String minWindow(String s, String t) {
    int[] chars = new int[128];
    boolean[] flag = new boolean[128];
    
    //先统计t中的字符情况
    for(int i = 0; i &lt; t.length(); i++){
        flag[t.charAt(i)] = true;
        chars[t.charAt(i)]++;
    }
    
    //移动滑动窗口，不断更改统计数据
    int cnt = 0, l = 0, min_l = 0, min_size = s.length() + 1;
    for(int r = 0; r &lt; s.length(); r++){
        if(flag[s.charAt(r)]){
            chars[s.charAt(r)]--;
            if(chars[s.charAt(r)] &gt;= 0){
                cnt++;
            }
            
            //若目前滑动窗口已包含t中全部字符
            //则尝试将1右移，在不影响结果的情况下活的最短子字符串
            while(cnt == t.length()){
                if(r - l + 1 &lt; min_size){
                    min_l = l;
                    min_size = r - l + 1;
                }
                
                chars[s.charAt(l)]++;
                if(flag[s.charAt(l)] &amp;&amp; chars[s.charAt(l)] &gt; 0){
                    cnt--;
                }
                l++;
            }
        }
    }
    //substring() returns the substring from start index to end index
    return min_size &gt; s.length() ? "" : s.substring(min_l, min_l + min_size);
}
</code></pre>
<h2 id="第四章----居合斩！二分查找">第四章 – 居合斩！二分查找</h2>
<h4 id="sqrtx">69. Sqrt(x)</h4>
<pre><code>public int mySqrt(int x) {
    if(x == 0) return x;
    int l = 1, r = x, mid, sqrt;
    while(l &lt;= r){
        //why not use "mid = (l + r) / 2;" is explained  in the post
        //https://stackoverflow.com/questions/6735259/calculating-mid-in-binary-search
        mid = l + (r - l) / 2;
        sqrt = x / mid;
    
        if(sqrt == mid){
            return mid;
        }
        else if(mid &gt; sqrt){
            r = mid - 1;
        }
        else{
            l = mid + 1;
        }
    }
    return r;
}
</code></pre>
<p>牛顿迭代法：</p>
<pre><code>public int mySqrt(int x) {
    long num = x;
    while(num * num &gt; x){
        num = (num + x / num) / 2;
    }
    return (int) num;
}
</code></pre>
<h4 id="find-first-and-last-position-of-element-in-sorted-array">34. Find First and Last Position of Element in Sorted Array</h4>
<pre><code>public int[] searchRange(int[] nums, int target) {
    //左闭右开
    if(nums.length == 0) return new int[]{-1, -1};
    
    int lower = lower_bound(nums, target);
    int upper = upper_bound(nums, target) - 1;
    
    if(lower == nums.length || nums[lower] != target) return new int[]{-1, -1};
    
    return new int[]{lower, upper};
}

private int lower_bound(int[] nums, int target){
    int l = 0, r = nums.length, mid;
    while(l &lt; r){
        mid = l + (r - l) / 2;
        if(nums[mid] &gt;= target){
            r = mid;
        }
        else{
            l = mid + 1;
        }
    }
    return l;
}
private int upper_bound(int[] nums, int target){
    int l = 0, r = nums.length, mid;
    while(l &lt; r){
        mid = l + (r - l) / 2;
        if(nums[mid] &gt; target){
            r = mid;
        }
        else{
            l = mid + 1;
        }
    }
    return l;
}
</code></pre>
<h4 id="search-in-rotated-sorted-array-ii">81. Search in Rotated Sorted Array II</h4>
<pre><code>public boolean search(int[] nums, int target) {
    int l = 0, r = nums.length - 1, mid;
    
    //这里用的是双闭区间
    while(l &lt;= r){
        mid = l + (r - l) / 2;
        if(nums[mid] == target){
            return true;
        }
        if(nums[l] == nums[mid]){
            l++;
        }
        else if(nums[mid] &lt;= nums[r]){
            if(target &gt; nums[mid] &amp;&amp; target &lt;= nums[r]){
                l = mid + 1;
            }
            else{
                r = mid - 1;
            }
        }
        else{
            if(target &gt;= nums[l] &amp;&amp; target &lt; nums[mid]){
                r = mid -1;
            }
            else{
                l = mid + 1;
            }
        }   
    }
    return false;
}
</code></pre>
<h2 id="第五章----千奇百怪的排序算法">第五章 – 千奇百怪的排序算法</h2>
<h4 id="kth-largest-element-in-an-array">215. Kth Largest Element in an Array</h4>
<pre><code>public int findKthLargest(int[] nums, int k) {
    int l = 0, r = nums.length - 1, target = nums.length - k;
    while(l &lt; r){
        int mid = quickSelection(nums, l, r);
        if(mid == target){
            return nums[mid];
        }
        if(mid &lt; target){
            l = mid + 1;
        }
        else{
            r = mid - 1;
        }
    }
    return nums[l];
}

private int quickSelection(int[] nums, int l, int r){
    int i = l + 1, j = r, temp;
    while(true){
        while(i &lt; r &amp;&amp; nums[i] &lt;= nums[l]){
            i++;
        }
        while(l &lt; j &amp;&amp; nums[j] &gt;= nums[l]){
            j--;
        }
        if(i &gt;= j){
            break;
        }
        temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    temp = nums[l];
    nums[l] = nums[j];
    nums[j] = temp;
    return j;
}
</code></pre>
<p>下面这个是用了Hoare’s Partition的快选，是leetcode题解的第二个解法，写起来挺麻烦的，是教科书算法。 贴在这里方便对比参考。</p>
<pre><code>int [] nums;

public void swap(int a, int b) {
    int tmp = this.nums[a];
    this.nums[a] = this.nums[b];
    this.nums[b] = tmp;
}

public int partition(int left, int right, int pivot_index) {
    int pivot = this.nums[pivot_index];
    // 1. move pivot to end
    swap(pivot_index, right);
    int store_index = left;

    // 2. move all smaller elements to the left
    for (int i = left; i &lt;= right; i++) {
        if (this.nums[i] &lt; pivot) {
            swap(store_index, i);
            store_index++;        
        }
    }

    // 3. move pivot to its final place
    swap(store_index, right);

    return store_index;
}

public int quickselect(int left, int right, int k_smallest) {
/*
Returns the k-th smallest element of list within left..right.
*/

if (left == right) // If the list contains only one element,
  return this.nums[left];  // return that element

// select a random pivot_index
Random random_num = new Random();
int pivot_index = left + random_num.nextInt(right - left); 

pivot_index = partition(left, right, pivot_index);

// the pivot is on (N - k)th smallest position
if (k_smallest == pivot_index)
  return this.nums[k_smallest];
// go left side
else if (k_smallest &lt; pivot_index)
  return quickselect(left, pivot_index - 1, k_smallest);
// go right side
return quickselect(pivot_index + 1, right, k_smallest);
}

public int findKthLargest(int[] nums, int k) {
    this.nums = nums;
    int size = nums.length;
    // kth largest is (N - k)th smallest
    return quickselect(0, size - 1, size - k);
}
</code></pre>
<h4 id="top-k-frequent-elements">347. Top K Frequent Elements</h4>
<p>这里也可以把nums.length替换成C++版本的那个max_count(最高频率)，我试了一下，由于更慢我就没用了</p>
<pre><code>public int[] topKFrequent(int[] nums, int k) {
    Map&lt;Integer, Integer&gt; counts = new HashMap&lt;&gt;();
    for(int num : nums){
        counts.put(num, counts.getOrDefault(num, 0) + 1);
    }
    
    List&lt;Integer&gt;[] bucket = new List[nums.length + 1];
    
    for(int key : counts.keySet()){
        int i = counts.get(key);
        if(bucket[i] == null){
            bucket[i] = new ArrayList&lt;&gt;();
        }
        bucket[i].add(key);
    }
    
    List&lt;Integer&gt; list = new ArrayList&lt;&gt;();
    for(int i = nums.length; i &gt;= 0 &amp;&amp; list.size() &lt; k; i--){
        if(bucket[i] != null) list.addAll(bucket[i]);
    }
    
    int[] ret = new int[list.size()];
    int i = 0;
    for (Integer e : list) ret[i++] = e;
    return ret;
}
</code></pre>
<h2 id="第六章----一切皆可搜">第六章 – 一切皆可搜</h2>
<h4 id="max-area-of-island">695. Max Area of Island</h4>
<p>用stack实现dfs</p>
<pre><code>int[] direction = {-1, 0, 1, 0, -1};
public int maxAreaOfIsland(int[][] grid) {
    int m = grid.length, n = grid[0].length, local_area = 0, area = 0, x, y;
    for(int i = 0; i &lt; m; i++){
        for(int j = 0; j &lt; n; j++){
            if(grid[i][j] == 1){
                local_area = 1;
                grid[i][j] = 0;
                Stack&lt;Pair&lt;Integer, Integer&gt;&gt; island = new Stack();
                island.push(new Pair&lt;&gt;(i, j));
                while(!island.isEmpty()){
                    Pair&lt;Integer, Integer&gt; pair = island.pop();
                    for(int k = 0; k &lt; 4; k++){
                        x = pair.getKey() + direction[k];
                        y = pair.getValue() + direction[k + 1];
                        if(x &gt;= 0 &amp;&amp; x &lt; m &amp;&amp; y &gt;= 0 &amp;&amp; y &lt; n &amp;&amp; grid[x][y] == 1){
                           grid[x][y] = 0;
                           local_area++;
                           island.push(new Pair&lt;&gt;(x, y));
                        }
                    }     
                }
                area = Math.max(area, local_area);
            }
        }
    }
    return area;
}
</code></pre>
<p>用递归遍历四个方向，个人比较喜欢用递归写dfs的逻辑</p>
<pre><code>public int maxAreaOfIsland(int[][] grid) {
    int max_area = 0;
    for(int i = 0; i &lt; grid.length; i++){
        for(int j = 0; j &lt; grid[0].length; j++){
            if(grid[i][j] == 1){
                max_area = Math.max(max_area, dfs(grid, i, j));
            }
        }
    }
    return max_area;
}

private int dfs(int[][] grid, int r, int c){
    if(r &gt;= grid.length || r &lt; 0 || c &gt;= grid[0].length || c &lt; 0 ||
       grid[r][c] == 0){
        return 0;
    }
    grid[r][c] = 0;
    return dfs(grid, r + 1, c) + dfs(grid, r - 1, c) +
           dfs(grid, r, c + 1) + dfs(grid, r, c - 1) + 1;
}
</code></pre>
<h4 id="number-of-provinces">547. Number of Provinces</h4>
<p>这题名字变了，以前叫Friend Circles， 要注意。<br>
这题可以不设visited，直接把isConnected中的访问过的1变成0起同样效果。每次遍历从isConnected[i][i]开始</p>
<pre><code>public int findCircleNum(int[][] isConnected) {
    int count = 0;
    int[] visited = new int[isConnected.length];
    for(int i = 0; i &lt; isConnected.length; i++){
        if(visited[i] == 0){
            dfs(isConnected, visited, i);
            count++;
        }
    }
    return count;
}

private void dfs(int[][] isConnected, int[] visited, int i){
    visited[i] = 1;
    for(int j = 0; j &lt; isConnected[0].length; j++){
        if(isConnected[i][j] == 1 &amp;&amp; visited[j] == 0){
            dfs(isConnected, visited, j);
        }
    }
}
</code></pre>
<h4 id="pacific-atlantic-water-flow">417. Pacific Atlantic Water Flow</h4>
<pre><code>int[] direction = {-1, 0, 1, 0, -1};
public List&lt;List&lt;Integer&gt;&gt; pacificAtlantic(int[][] heights) {
    List&lt;List&lt;Integer&gt;&gt; ans = new LinkedList&lt;&gt;();
    
    if(heights == null || heights.length == 0 || heights[0].length == 0) return ans;
    
    int m = heights.length, n = heights[0].length;
    boolean[][] can_reach_p = new boolean[m][n];
    boolean[][] can_reach_a = new boolean[m][n];
    
    for(int i = 0; i &lt; m; i++){
        dfs(heights, can_reach_p, i, 0);
        dfs(heights, can_reach_a, i, n - 1);
    }
    
    for(int i = 0; i &lt; n; i++){
        dfs(heights, can_reach_p, 0, i);
        dfs(heights, can_reach_a, m - 1, i);
    }
    
    for(int i = 0; i &lt; m; i++){
        for(int j = 0; j &lt; n; j++){
         if(can_reach_p[i][j] &amp;&amp; can_reach_a[i][j]){
             //ans.add(List.of(i,j));
             ans.add(Arrays.asList(i, j));
         }   
        }
    }
    return ans;
}

private void dfs(int[][] heights, boolean[][] can_reach, int r, int c){
    if(can_reach[r][c]){
        return ;
    }
    can_reach[r][c] = true;
    int x, y;
    for(int i = 0; i &lt; 4; i++){
        x = r + direction[i];
        y = c + direction[i + 1];
        if(x &gt;= 0 &amp;&amp; x &lt; heights.length &amp;&amp; y &gt;= 0 &amp;&amp; y &lt; heights[0].length &amp;&amp; 
           heights[r][c] &lt;= heights[x][y]){
            dfs(heights, can_reach, x, y);
        }
    }
}
</code></pre>
<h4 id="permutations">46. Permutations</h4>
<pre><code>public List&lt;List&lt;Integer&gt;&gt; permute(int[] nums) {
    List&lt;List&lt;Integer&gt;&gt; ans = new LinkedList&lt;&gt;();
    
    ArrayList&lt;Integer&gt; nums_list = new ArrayList&lt;Integer&gt;();
    for (int num : nums) nums_list.add(num);
    
    backtracking(nums_list, 0, ans);
    return ans;
}

private void backtracking(ArrayList&lt;Integer&gt; nums, int level, List&lt;List&lt;Integer&gt;&gt; ans){
    if(level == nums.size() - 1){
        ans.add(new ArrayList&lt;Integer&gt;(nums));
        return;
    }
    
    for(int i = level; i &lt; nums.size(); i++){
        Collections.swap(nums, i, level);
        backtracking(nums, level + 1, ans);
        Collections.swap(nums, i, level);
    }
}


public List&lt;List&lt;Integer&gt;&gt; combine(int n, int k) {
    List&lt;List&lt;Integer&gt;&gt; ans = new LinkedList&lt;&gt;();
    
    int[] comb = new int[k];
    int count = 0;
    
    backtracking(ans, comb, count, 1, n, k);
    return ans;
}
</code></pre>
<h4 id="combinations">77. Combinations</h4>
<pre><code>private void backtracking(List&lt;List&lt;Integer&gt;&gt; ans, int[] comb, int count, int pos, int n, int k){
    if(count == k){
        List&lt;Integer&gt; a = new ArrayList&lt;&gt;();
        for(int c : comb) a.add(c);
        ans.add(a);
        return ;
    }
    for(int i = pos; i &lt;= n; i++){
        comb[count++] = i;
        backtracking(ans, comb, count, i + 1, n, k);
        count--;
    }
}
</code></pre>
<p>或者是这样写，Java处理array和list之间的转换比较麻烦</p>
<pre><code>public List&lt;List&lt;Integer&gt;&gt; combine(int n, int k) {
    List&lt;List&lt;Integer&gt;&gt; ans = new ArrayList();
    backTracking(ans, new ArrayList(), 1, n, k);    
    return ans;
}

 public void backTracking(List&lt;List&lt;Integer&gt;&gt; ans, List&lt;Integer&gt; comb, int pos, int n, int k){
     if(comb.size() == k){
         ans.add(new ArrayList(comb));
     }

     for(int i = pos; i &lt;= n; ++i){
            comb.add(i);
            backTracking(ans, comb, i + 1, n, k);
            comb.remove(comb.size() - 1);
        }
 }
</code></pre>
<p>又或者这样</p>
<pre><code>public List&lt;List&lt;Integer&gt;&gt; combine(int n, int k) {
    List&lt;List&lt;Integer&gt;&gt; ans = new LinkedList&lt;&gt;();
    
    List&lt;Integer&gt; comb = new ArrayList&lt;&gt;();
    int count = 0;
    
    backtracking(ans, comb, count, 1, n, k);
    return ans;
}

private void backtracking(List&lt;List&lt;Integer&gt;&gt; ans, List&lt;Integer&gt; comb, int count, int pos, int n, int k){
    if(count == k){
        ans.add(new ArrayList&lt;Integer&gt;(comb));
        return ;
    }
    for(int i = pos; i &lt;= n; i++){
        comb.add(i);
        count++;
        backtracking(ans, comb, count, i + 1, n, k);
        comb.remove(--count);
    }
}
</code></pre>
<h4 id="word-search">79. Word Search</h4>
<pre><code>public boolean exist(char[][] board, String word) {
    if(board.length == 0) return false;
    int m = board.length, n = board[0].length;
    boolean[][] visited = new boolean[m][n];
    boolean[] find = new boolean[]{false};
    
    for(int i = 0; i &lt; m; i++){
        for(int j = 0; j &lt; n; j++){
            backtracking(i, j, board, word, find, visited, 0);
        }
    }
    return find[0];
}

private void backtracking(int i, int j, char[][] board, String word, boolean[] find, boolean[][] visited, int pos){
    if(i &lt; 0 || i &gt;= board.length || j &lt; 0 || j &gt;= board[0].length){
        return ;
    }
    
    if(visited[i][j] || find[0] == true || board[i][j] != word.charAt(pos)){
        return ;
    }
    
    if(pos == word.length() - 1){
        find[0] = true;
        return ;
    }
    
    visited[i][j] = true;
    //递归子节点
    backtracking(i + 1, j, board, word, find, visited, pos + 1);
    backtracking(i - 1, j, board, word, find, visited, pos + 1);
    backtracking(i, j + 1, board, word, find, visited, pos + 1);
    backtracking(i, j - 1, board, word, find, visited, pos + 1);
    visited[i][j] = false;
}
</code></pre>
<p>或者这样写</p>
<pre><code>public boolean exist(char[][] board, String word) {
    for(int i = 0; i &lt; board.length; i ++){
        for(int j = 0; j &lt; board[0].length; j++){
            if(word.charAt(0) == board[i][j] &amp;&amp; backtracking(i, j, 0, word, board)){
                return true;
            }
        }
    }
    return false;
}

public boolean backtracking(int i, int j, int index, String word, char[][] board){
    if(index == word.length()) return true;
    
    if(i &lt; 0 || i &gt;= board.length || 
       j &lt; 0 || j &gt;= board[i].length || 
       word.charAt(index) != board[i][j] || board[i][j] == '0'){
        return false;
    }
    char temp = board[i][j];
    board[i][j] = '0';
    if(backtracking(i + 1, j, index + 1, word, board) ||
       backtracking(i - 1, j, index + 1, word, board) ||
       backtracking(i, j + 1, index + 1, word, board) ||
       backtracking(i, j - 1, index + 1, word, board)){
        return true;
    }
    board[i][j] = temp;
    return false;
}
</code></pre>
<h4 id="n-queens">51. N-Queens</h4>
<pre><code>public List&lt;List&lt;String&gt;&gt; solveNQueens(int n) {
    List&lt;List&lt;String&gt;&gt; ans = new LinkedList&lt;&gt;();
    if(n == 0){
        return ans;
    }
    
    char[][] board = new char[n][n];
    for(int i = 0; i &lt; n; i++){
        for(int j = 0; j &lt; n; j++){
            board[i][j] = '.';
        }
    }
    
    boolean[] column = new boolean[n];
    boolean[] ldiag = new boolean[2 * n - 1];
    boolean[] rdiag = new boolean[2 * n - 1];
    backtracking(ans, board, column, ldiag, rdiag, 0, n);
    return ans;
}

private void backtracking(List&lt;List&lt;String&gt;&gt; ans,  char[][] board, boolean[] column, boolean[] ldiag, boolean[] rdiag, int row, int n){
    if(row == n){
        List&lt;String&gt; str = new LinkedList&lt;&gt;();
        for(int i = 0; i &lt; n; i++){
            StringBuilder sb = new StringBuilder();
            for(int j = 0; j &lt; n; j++){
                sb.append(board[i][j]);
            }
            str.add(sb.toString());
        }
        ans.add(new LinkedList&lt;String&gt;(str));
        return ;
    }
    
    for(int i = 0; i &lt; n; i++){
        if(column[i] || ldiag[n - row + i - 1] || rdiag[row + i]){
            continue;
        }
        
        board[row][i] = 'Q';
        column[i] = ldiag[n - row + i - 1] = rdiag[row + i] = true;
        backtracking(ans, board, column, ldiag, rdiag, row + 1, n);
        board[row][i] = '.';
        column[i] = ldiag[n - row + i - 1] = rdiag[row + i] = false;
    }
}
</code></pre>
<h4 id="shortest-bridge">934. Shortest Bridge</h4>
<pre><code>private int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

public int shortestBridge(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    int bridge_distance = 0;
    
    Queue&lt;int[]&gt; queue = new LinkedList&lt;&gt;();
    boolean[][] visited = new boolean[m][n];
    boolean first_island_found = false;
    
    // Find the land coverage of 1st island using DFS
    for(int i = 0; i &lt; m; i++)
    {
        if(first_island_found) break;
        for(int j = 0; j &lt; n; j++)
        {
            if(grid[i][j] == 1)
            {
                find_first_land(grid, i, j, m, n, queue, visited);
                first_island_found = true;
                break;
            }
        }
    }
    
    // Find the shortest distance to build bridge between 1st bridge and 2nd bridge using BFS
    while(!queue.isEmpty())
    {
        int size = queue.size();
        
        while(size &gt; 0)
        {
            int[] popped = queue.remove();
            
            for(int[] direction : directions)
            {
                int x = popped[0] + direction[0];
                int y = popped[1] + direction[1];
                
                if(x &gt; -1 &amp;&amp; y &gt; -1 &amp;&amp; x &lt; m &amp;&amp; y &lt; n &amp;&amp; !visited[x][y])
                {
                    if(grid[x][y] == 1)
                    {
                        return bridge_distance;
                    }
                    
                    queue.add(new int[] {x, y});
                    visited[x][y] = true;
                }
            }
            size--;
        }
        bridge_distance++;
    }
    return -1;
}

public void find_first_land(int[][] grid, int i, int j, int m, int n, Queue&lt;int[]&gt; queue, boolean[][] visited)
{
    if(i &lt; 0 || i &gt;= m || j &lt; 0 || j &gt;= n || grid[i][j] == 0 || visited[i][j]) return;
    
    queue.add(new int[] {i, j});
    visited[i][j] = true;
    
    find_first_land(grid, i-1, j, m, n, queue, visited);
    find_first_land(grid, i+1, j, m, n, queue, visited);
    find_first_land(grid, i, j-1, m, n, queue, visited);
    find_first_land(grid, i, j+1, m, n, queue, visited);
}
</code></pre>
<h2 id="第七章----深入浅出的动态规划">第七章 – 深入浅出的动态规划</h2>
<h4 id="climbing-stairs">70. Climbing Stairs</h4>
<p>用dp</p>
<pre><code>public int climbStairs(int n) {
    if(n &lt;= 2) return n;
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = 1;
    for(int i = 2; i &lt;= n; i++){
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
</code></pre>
<p>不用dp</p>
<pre><code>public int climbStairs(int n) {
    if(n &lt; 3) return n;
    int pre2 = 1, pre1 = 2, cur = 0;
    for(int i = 2; i &lt; n; i++){
        cur = pre1 + pre2;
        pre2 = pre1;
        pre1 = cur;
    }
    return cur;
}
</code></pre>
<h4 id="house-robber">198. House Robber</h4>
<pre><code>    if(nums.length == 0) return 0;
    int n = nums.length;
    int[] dp = new int[n + 1];
    dp[1] = nums[0];
    for(int i = 2; i &lt;= n; i++){
        dp[i] = Math.max(dp[i - 1], nums[i - 1] + dp[i - 2]);
    }
    return dp[n];
}
</code></pre>
<p>还是不用dp的方法</p>
<pre><code>public int rob(int[] nums) {
    if(nums.length == 0) return 0;
    int n = nums.length;
    if(n == 1) return nums[0];
    int pre2 = 0, pre1 = 0, cur = 0;
    for(int i = 0; i &lt; n; i++){
        cur = Math.max(pre2 + nums[i], pre1);
        pre2 = pre1;
        pre1 = cur;
    }
    return cur;
}
</code></pre>
<h4 id="arithmetic-slices">413. Arithmetic Slices</h4>
<pre><code>public int numberOfArithmeticSlices(int[] nums) {
    if(nums.length &lt; 3) return 0;
    int[] dp = new  int[nums.length];
    for(int i = 2; i &lt; nums.length; i++){
        if(nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]){
            dp[i] = dp[i - 1] + 1;
        }
    }
    
    int result = 0;
    for(int i = 0; i &lt; dp.length; i++){
        result += dp[i];
    }
    return result;
}
</code></pre>
<h4 id="minimum-path-sum">64. Minimum Path Sum</h4>
<pre><code>public int minPathSum(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    int[][] dp = new int[m][n];
    for(int i = 0; i &lt; m; i++){
        for(int j = 0; j &lt; n; j++){
            if(i == 0 &amp;&amp; j == 0){
                dp[i][j] = grid[i][j];
            } else if(i == 0){
                dp[i][j] = dp[i][j - 1] + grid[i][j];
            } else if(j == 0){
                dp[i][j] = dp[i - 1][j] + grid[i][j];
            } else{
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
    }
    return dp[m - 1][n - 1];
}
</code></pre>
<h4 id="matrix">542. 01 Matrix</h4>
<pre><code>public int[][] updateMatrix(int[][] mat) {
    int n = mat.length;
    int m = mat[0].length;
    int[][] dp = new int[n][m];
    
    for(int i = 0; i &lt; n; i++){
        for(int j = 0; j &lt; m; j++){
            dp[i][j] = Integer.MAX_VALUE - 1;
        }
    }
    
    for(int i = 0; i &lt; n; i++){
        for(int j = 0; j &lt; m; j++){
            if(mat[i][j] == 0){
                dp[i][j] = 0;
            }
            else{
                if(j &gt; 0){
                    dp[i][j] = Math.min(dp[i][j], dp[i][j - 1] + 1);
                }
                if(i &gt; 0){
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][j] + 1);
                }
            }
        }
    }
    
    for(int i = n - 1; i &gt;= 0; i--){
        for(int j = m - 1; j &gt;= 0; j--){
            if(mat[i][j] != 0){
                if(j &lt; m - 1){
                    dp[i][j] = Math.min(dp[i][j], dp[i][j + 1] + 1);
                }
                if(i &lt; n - 1){
                    dp[i][j] = Math.min(dp[i][j], dp[i + 1][j] + 1);
                }
            }
        }
    }
    return dp;
}
</code></pre>
<h4 id="maximal-square">221. Maximal Square</h4>
<pre><code>public int maximalSquare(char[][] matrix) {
    int rows = matrix.length, cols = rows &gt; 0 ? matrix[0].length : 0;
    int[][] dp = new int[rows + 1][cols + 1];
    int maxsqlen = 0;
    for (int i = 1; i &lt;= rows; i++) {
        for (int j = 1; j &lt;= cols; j++) {
            if (matrix[i-1][j-1] == '1'){
                dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
                maxsqlen = Math.max(maxsqlen, dp[i][j]);
            }
        }
    }
    return maxsqlen * maxsqlen;
}
</code></pre>
<h4 id="perfect-squares">279. Perfect Squares</h4>
<pre><code>public int numSquares(int n) {
    int[] memo = new int[n + 1];

    memo[0] = 0;
    for(int i = 1; i &lt;= n; i++){
        memo[i] = Integer.MAX_VALUE;
        for(int j = 1; j * j &lt;= i; j++){
            memo[i] = Math.min(memo[i], memo[i - j * j] + 1);
        }
    }
    return memo[n];
}
</code></pre>
<h2 id="第十三章----指针三剑客之一：-链表">第十三章 – 指针三剑客之一： 链表</h2>
<h4 id="reverse-linked-list">206. Reverse Linked List</h4>
<pre><code>public ListNode reverseList(ListNode head) {
    return reverseList(head, null);
}
private ListNode reverseList(ListNode head, ListNode prev){
    if(head == null) return prev;
    ListNode next = head.next;
    head.next = prev;
    return reverseList(next, head);
}
</code></pre>
<p>不用递归</p>
<pre><code>public ListNode reverseList(ListNode head) {
    ListNode prev = null, next;
    while(head != null){
        next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
</code></pre>
<h4 id="merge-two-sorted-lists">21. Merge Two Sorted Lists</h4>
<p>递归</p>
<pre><code>public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if(l2 == null) return l1;
    if(l1 == null) return l2;
    if(l1.val &gt; l2.val){
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
    }
    l1.next = mergeTwoLists(l1.next, l2);
    return l1;
}
</code></pre>
<p>非递归</p>
<pre><code>public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0), node = dummy;
    while(l1 != null &amp;&amp; l2 != null){
        if(l1.val &lt;= l2.val){
           node.next = l1;
            l1 = l1.next; 
        }
        else{
            node.next = l2;
            l2 = l2.next;
        }
        node = node.next;
    }
            
    if(l1 == null &amp;&amp; l2 != null) node.next = l2;
    else node.next = l1;
    return dummy.next;
}
</code></pre>
<h4 id="swap-nodes-in-pairs">24. Swap Nodes in Pairs</h4>
<pre><code>public ListNode swapPairs(ListNode head) {
    ListNode p = head, s;
    if(p != null &amp;&amp; p.next != null){
        s = p.next;
        p.next = s.next;
        s.next = p;
        head = s;
        while(p.next != null &amp;&amp; p.next.next != null){
            s = p.next.next;
            p.next.next = s.next;
            s.next = p.next;
            p.next = s;
            p = s.next;
        }
    }
    return head;
}
</code></pre>
<p>简单点的写法</p>
<pre><code>public ListNode swapPairs(ListNode head) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    head = dummy;
    while(head.next != null &amp;&amp; head.next.next != null){
        ListNode n1 = head.next;
        ListNode n2 = head.next.next;
        n1.next = n2.next;
        n2.next = n1;
        //head always hold one node before the first and the second node
        head.next = n2;
        head = n1;
    }
    return dummy.next;
}
</code></pre>
<h4 id="intersection-of-two-linked-lists">160. Intersection of Two Linked Lists</h4>
<pre><code>public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode l1 = headA, l2 = headB;
    while(l1 != l2){
        if(l1 == null) l1 = headB;
        else l1 = l1.next;
        if(l2 == null) l2 = headA;
        else l2 = l2.next;
    }
    return l1;
}
</code></pre>
<h4 id="palindrome-linked-list">234. Palindrome Linked List</h4>
<pre><code>public boolean isPalindrome(ListNode head) {
    if(head == null || head.next == null) return true;
    ListNode slow = head, fast = head;
    while(fast.next != null &amp;&amp; fast.next.next != null){
        slow = slow.next;
        fast = fast.next.next;
    }
    slow.next = reverseList(slow.next);
    slow = slow.next;
    while(slow != null){
        if(head.val != slow.val) return false;
        head = head.next;
        slow = slow.next;
    }
    return true;
}

private ListNode reverseList(ListNode head){
    ListNode prev = null, next;
    while(head != null){
        next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
</code></pre>
<h2 id="第十四章----指针三剑客之一：-树">第十四章 – 指针三剑客之一： 树</h2>
<h4 id="maximum-depth-of-binary-tree">104. Maximum Depth of Binary Tree</h4>
<pre><code>public int maxDepth(TreeNode root) {
    if(root == null){
        return 0;
    }
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
}
</code></pre>
<h4 id="balanced-binary-tree">110. Balanced Binary Tree</h4>
<pre><code>public boolean isBalanced(TreeNode root) {
    if(backtrack(root) != -1){
        return true;
    }
    return false;
}

private int backtrack(TreeNode root){
    if(root == null){
        return 0;
    }
    
    int left = backtrack(root.left);
    int right = backtrack(root.right);
    
    if(left == -1 || right == -1 || Math.abs(right - left) &gt; 1){
        return -1;
    }
    return 1 + Math.max(left, right);
}
</code></pre>
<h4 id="diameter-of-binary-tree">543. Diameter of Binary Tree</h4>
<pre><code>int diameter = 0;
public int diameterOfBinaryTree(TreeNode root) {
    backtrack(root);
    return diameter;
}

private int backtrack(TreeNode root){
    if(root == null){
        return 0;
    }
    
    int l = backtrack(root.left);
    int r = backtrack(root.right);
    
    diameter = Math.max(diameter, l + r);
    
    return Math.max(l, r) + 1;
}
</code></pre>
<h4 id="path-sum-iii">437. Path Sum III</h4>
<pre><code>int diameter = 0;
public int diameterOfBinaryTree(TreeNode root) {
    backtrack(root);
    return diameter;
}

private int backtrack(TreeNode root){
    if(root == null){
        return 0;
    }
    
    int l = backtrack(root.left);
    int r = backtrack(root.right);
    
    diameter = Math.max(diameter, l + r);
    
    return Math.max(l, r) + 1;
}
</code></pre>
<h4 id="symmetric-tree">101. Symmetric Tree</h4>
<pre><code>public boolean isSymmetric(TreeNode root) {
    if(root == null){
        return false;
    }
    
    return isSymmetric(root.left, root.right);
}

private boolean isSymmetric(TreeNode left, TreeNode right){
    if(left == null &amp;&amp; right == null){
        return true;
    }
    if(left == null || right == null){
        return false;
    }
    if(left.val != right.val){
        return false;
    }
    return isSymmetric(left.right, right.left) &amp;&amp;
           isSymmetric(left.left, right.right);
}
</code></pre>
<h4 id="delete-nodes-and-return-forest">1110. Delete Nodes And Return Forest</h4>
<pre><code>List&lt;TreeNode&gt; result = new LinkedList&lt;&gt;();
Set&lt;Integer&gt; delete = new HashSet&lt;&gt;();
public List&lt;TreeNode&gt; delNodes(TreeNode root, int[] to_delete) {
    for(int i : to_delete) delete.add(i);
    
    root = helper(root);
    if(root != null){
        result.add(root);
    }
    return result;
}
private TreeNode helper(TreeNode root){
    if(root == null){
        return root;
    }
    
    root.left = helper(root.left);
    root.right = helper(root.right);
    if(delete.contains(root.val)){
        if(root.left != null) result.add(root.left);
        if(root.right != null) result.add(root.right);
        root = null;
    }
    return root;
}
</code></pre>
<h4 id="average-of-levels-in-binary-tree">637. Average of Levels in Binary Tree</h4>
<pre><code>public List&lt;Double&gt; averageOfLevels(TreeNode root) {
    List&lt;Double&gt; result = new LinkedList&lt;&gt;();
    if(root == null){
        return result;
    }
    
    Queue&lt;TreeNode&gt; q = new LinkedList&lt;&gt;();
    q.offer(root);
    while(!q.isEmpty()){
        int level = q.size();
        double sum = 0;
        for(int i = 0; i &lt; level; i++){
            TreeNode node = q.poll();
            sum += node.val;
            
            if(node.left != null){
                q.add(node.left);
            }                
            if(node.right != null){
                q.add(node.right);
            }
        }
        
        result.add(sum / level);
    }
    return result;
}
</code></pre>
<h4 id="construct-binary-tree-from-preorder-and-inorder-traversal">105. Construct Binary Tree from Preorder and Inorder Traversal</h4>
<pre><code>public TreeNode buildTree(int[] preorder, int[] inorder) {
    if(preorder.length == 0) return null;
    
    Map&lt;Integer, Integer&gt; hash = new HashMap&lt;&gt;();
    for(int i = 0; i &lt; preorder.length; i++){
        hash.put(inorder[i], i);
    }
    return buildTreeHelper(hash, preorder, 0, preorder.length - 1, 0);
}
private TreeNode buildTreeHelper(Map&lt;Integer, Integer&gt; hash, int[] preorder, int s0, int e0, int s1){
    if(s0 &gt; e0) return null;
    int mid = preorder[s1], index = hash.get(mid), leftLen = index - s0 - 1;
    TreeNode node = new TreeNode(mid);
    node.left = buildTreeHelper(hash, preorder, s0, index - 1, s1 + 1);
    node.right = buildTreeHelper(hash, preorder, index + 1, e0, s1 + 2 + leftLen);
    return node;
}
</code></pre>

