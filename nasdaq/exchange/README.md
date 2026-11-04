# 나스닥 연구창 ↔ 시간외창 교환 폴더

브랜치: `ccr-4dc2bcfa-19dlf6` (저장소 redich04011-maker/dash)

| 방향 | 파일 이름 규칙 | 내용 |
|---|---|---|
| 시간외 → 연구 | `from_local/YYYYMMDD_주제.csv` / `.md` | 판에 붙여 잰 결과, 우리 밤 성과 갱신, 요청 |
| 연구 → 시간외 | `from_research/YYYYMMDD_주제.csv` / `.md` | 예측 깃발, 분석 결과, 판정 |

절차
1. 파일을 올린 쪽은 `git push` 후 상대 세션에 send_message로 "올림: 경로"만 알린다.
2. 받은 쪽은 `git pull` 해서 읽고, 결과를 반대 폴더에 올린 뒤 같은 방식으로 알린다.
3. 사람이 복사·붙여넣기 하지 않는다.

세션
- 연구창(클라우드): session_01B1PKpz7tQuyMbwvawwRce2
- 시간외창(로컬): session_01B1PKq2hdEYuM9XYRfwREKb
