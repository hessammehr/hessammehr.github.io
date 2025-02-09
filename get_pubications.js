const pubs = Array.from(document.querySelectorAll('tr.gsc_a_tr'))
const citations = pubs.map((tr) => {
    const title = tr.querySelector('a.gsc_a_at').innerText
    // console.log(title)
    const year = parseInt(tr.querySelector('span.gsc_a_h').innerText) || 0
    if (year < 2010) return
    const [authors, place] = Array.from(tr.querySelectorAll('div.gs_gray')).map((div) => div.innerText)
    const formatted_authors = authors.split(',').map((author) => author.includes('Mehr') && '[Mehr, S. H. M.]{.underline}' || author)
    const [journal, rest] = place.match(/([^\d]+)(.*)/).slice(1).map(s => s.trim())
    return `
### ${title} 
${formatted_authors.join(', ')}  
*${journal}* ${rest} **${year}**
    `
})

copy(citations.join(''))