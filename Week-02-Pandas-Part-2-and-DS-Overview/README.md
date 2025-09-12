# Week 2 - Pandas Part 2

## Agenda 
0. Remind me to 
	* Start recording
	* Turn on CC's 
	* Enable screen sharing for everyone (for breakout rooms)
1. Announcements
    * Goolge Workshop on Data Privacy (CTP Fellow Exclusive).  [All info on slack here.](https://ctpc11.slack.com/archives/C095CMC2JGG/p1756821811976879)
    * [RSVP Here ASAP](https://forms.gle/ZrBuqaD3G9FkULK2A)
    * Weekly announcement every Tuesday in the channel [#announcements](https://ctpc11.slack.com/archives/C095308KR1P)
2. Best way to ask questions on Slack.
    * Include a full screen shot of your error message and the code to get the error.  
    * "What did you do to debug this already?"
        * *Directly from Meta*
3. Review HW.
4. [Google Slide Lecture](https://docs.google.com/presentation/d/1uoWIMjfH70CUrHKJnckMfd6ppQTxFwDKFlonHElwpTQ/edit)
5. BREAK
6. Finding Data Live Demo
    * [Google Doc of good data sources](https://docs.google.com/document/d/1VvmTmHrURfV24owFeew33S8INOLE9iNnRFXntSFhZdc/edit) <-- please contribute
7. Cleaning Data Lecture.ipynb
8. Github - How to sync fork.
9. Review what Is due for next week.


# Homework Due Thursday 09/11/2025
---

**_All HWs are due 1 day before the next class begins by 12:01 PM (Noon) - Thursday, September 11, 2025._**

_Submit them via the [HW Submssions Form](https://forms.gle/MFH173MZaQ5TquCB6)_

**DEADLINE: Thursday, September 11, 2025**

---

### 1. Pre-Class for Next Weeks Analytics Lecture & Slack Message

1. [Storytelling with data](https://www.youtube.com/watch?v=hVimVzgtD6w&ab_channel=TED) Ted (not TedX). [~20min]
2. [BBC: Joy of Stats](https://www.youtube.com/watch?v=jbkSRLYSojo&ab_channel=BBC) [~4min]
3. [Seaborn Tutorial](https://www.youtube.com/watch?v=LnGz20B3nTU&ab_channel=AbsentData) [~15min]
4. [Charts in Seaborn](https://www.youtube.com/watch?v=Iui04c3tbH8&ab_channel=FaniloAndrianasolo) [~4min]
5. Optional: [Joy of stats full documentary](https://www.gapminder.org/videos/the-joy-of-stats/) [~1hr]

After watching the following videos above... 
* Post image in this weeks slack thread with one data viz YOU loved or YOU hated and why you felt that way.
	* _(I'm looking for the story and impact it had on your life.)_
* Submit link to slack message in HW Submission Sheet.

---

### 2. Build your first dashboard. 
1. Use these Article + YouTube Tutorial + GitHub code on [How to build a dashboard in Streamlit](https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/) [~45 mins]
2. Submit the link to your DELPLOYED dashboard in the _HW Submission Sheet_
3. _Link should look something like this https://population-dashboard.streamlit.app/_

---

### 3. Exercise
1. MAKE A COPY of the Exercise file. 
2. Complete the questions. 
3. Create Pull Request
* Submit link to your exercise file in the HW Submssion Sheet.

---

### 4. LinkedIn Post
__Submit by putting the link to your LinkedIn post under the "LinkedIn Post" column.__

Yoru first LinkedIn post can be about something you found and like and is related to your field of interest.

* Post/Repost a video, article, paper or another LinkedIn post you liked and say/describe why you liked it. 
    * 'commenting' on another post does not count. 
    * you can repost instead.
    * Maybe do a poll.
    * __Tag people to make a greater imporession__
---
<br>

## Finding Data in the Wild

#### Finding Data your project. 
0. Here we are going to mock data for a project about dogs. 
1. Google dog dataset. [google it, find kaggle and data world and stanford and opennyc]
2. Go to the dataworld link, about 5 links down you'll see the dog [NYC Dog Licensing Dataset](https://data.world/city-of-ny/nu7n-tubp). 
3. Point out it was last updated 2021-07-29.
4. Then go the the source, and see the source was updated this year. 
5. In the source, point out the data dictionary and user guide, scroll down to see the column descriptons. 
6. Maybe sidetrack on doing one of those evolution trees (how to look how to make one of those evolution trees for dogs)
    * google dog evolution tree data
    * find research gate
    * see realted images
    * see the word 'phylogenetic', thats a new google term you can use. 
    * open paper, use command+f search for 'data' or 'download' 

#### How I got the data for this lecture?
0. Google, ["pandas data cleaning tutorial"]((https://www.google.com/search?q=pandas+data+cleaning+tutorial&oq=pandas+data+cleaning+tutorial+&gs_lcrp=EgZjaHJvbWUyCAgAEEUYHhg5Mg0IARAAGIYDGIAEGIoFMg0IAhAAGIYDGIAEGIoFMg0IAxAAGIYDGIAEGIoFMgoIBBAAGIAEGKIEMgoIBRAAGIAEGKIEMgoIBhAAGIAEGKIE0gEINTAzNWowajGoAgCwAgA&sourceid=chrome&ie=UTF-8))
1. Open a few links. But end up on the [w3 schools tutorial](https://www.w3schools.com/python/pandas/pandas_cleaning.asp). 
2. Try and get that data. 
    - I did, copy and paste into sublime and did removing whitespace tricks in there. 
    - Then copy and did pd.read_clipboard() which works very well. 
3. Now load that data into the notebook and flow into the data cleaning lecture. 
