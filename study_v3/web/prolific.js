'use strict';
// Recruitment entry is separate from public game demos. Never allocate on GET.
window.ProlificEntry = (() => {
 const reviewDomain=/^\/review\/(warehouse|pong|kitchen)\/$/.exec(location.pathname)?.[1]||null;
 const storageKey='policylens.review.invite.'+reviewDomain;
 let invitation=new URLSearchParams(location.hash.slice(1)).get('invite')||'';
 if(reviewDomain){try{if(invitation)sessionStorage.setItem(storageKey,invitation);else invitation=sessionStorage.getItem(storageKey)||'';}catch{} if(location.hash)history.replaceState(null,'',location.pathname);}
 const endpoint=reviewDomain?'/api/review/'+reviewDomain:'/api/prolific';
 const params = new URLSearchParams(location.search);
 const ids = Object.fromEntries(['PROLIFIC_PID','STUDY_ID','SESSION_ID'].map(k=>[k,params.get(k)||'']));
 const valid = reviewDomain?!!invitation:Object.values(ids).every(v=>/^[0-9a-f]{24}$/.test(v));
 const escape = s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 let info=null, decision='', enrolling=false, declined=false;
 async function load(){const r=await fetch(endpoint+'/info',{cache:'no-store'});if(!r.ok)throw new Error('entry_unavailable');info=await r.json();return info;}
 function render(){
  if(declined&&reviewDomain)return '<section class="panel consent-document"><h1>Preview not started.</h1><p>You may close this page. No game record has been created.</p></section>';
  if(declined)return '<section class="panel consent-document"><h1>You have not joined the study.</h1><p>No consent or game record has been submitted. You may close this page and return your submission on Prolific.</p><a class="button" href="https://app.prolific.com/">Return to Prolific</a></section>';
  if(!info)return '<section class="panel consent-document"><h1>Participant information</h1><p>We could not load the study information. Please refresh, or contact the researcher through Prolific.</p></section>';
  const c=info.consent;
  return `${reviewDomain?'<section class="panel notice"><strong>A-group researcher preview</strong><p>This uses the complete participant study flow, including the consent page below, tutorial, Tasks 1, 2 and 3, and survey. This review is unpaid, uses a separate preview session, and does not occupy a Prolific place.</p></section>':''}<section class="panel consent-document"><p><b>Project Title:</b> <i>${escape(c.title)}</i><br><b>NTU-IRB Ref No.:</b> ${escape(c.irb_reference)}</p>${!info.ready?'<p class="notice">This study is being prepared and is not yet accepting participants.</p>':''}${!valid?(reviewDomain?'<p class="notice">Please open the complete review invitation link supplied by the researcher.</p>':'<p class="notice">Please open this study using your study link on Prolific. The participant and submission identifiers are missing or invalid.</p>'):''}${c.paragraphs.map(p=>'<p>'+p+'</p>').join('')}<form id="prolificConsentForm"><label class="check-field"><input type="radio" name="decision" value="yes" ${decision==='yes'?'checked':''}><span>${escape(c.agreement)}</span></label><label class="check-field"><input type="radio" name="decision" value="no" ${decision==='no'?'checked':''}><span>${escape(c.decline)}</span></label><div class="button-row"><button id="prolificContinue" class="button" ${!info.ready||!valid||enrolling||decision!=='yes'?'disabled':''}>Consent and Continue</button><button id="prolificDecline" type="button" class="button secondary">I do not wish to participate</button><button id="prolificPrint" type="button" class="button secondary">Print / Save a copy</button></div><p id="prolificError" role="alert"></p></form></section>`;
 }
 function bind(onEnrol,redraw){
  const form=document.getElementById('prolificConsentForm');if(!form)return;
  form.querySelectorAll('[name="decision"]').forEach(input=>input.onchange=()=>{decision=input.value;redraw();});
  document.getElementById('prolificDecline').onclick=()=>{if(enrolling)return;declined=true;redraw();};
  document.getElementById('prolificPrint').onclick=()=>window.print();
  form.onsubmit=async e=>{
   e.preventDefault();if(enrolling||!info.ready||!valid||decision!=='yes')return;
   enrolling=true;document.getElementById('prolificContinue').disabled=true;
   try{
    const r=await fetch(endpoint+'/enrol',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({...ids,...(reviewDomain?{invitation}:{}),consent:true,age_21:true,consent_version:info.consent_version})});
    const result=await r.json();
    if(!r.ok){const messages={invalid_review_invitation:'This review invitation is invalid or expired. Please request a new link.',review_release_changed:'The study version has changed. Please request a new review link.',review_invitation_full:'This review link has reached its preview session limit. Please request a new link.',prolific_submission_closed:"This Prolific submission has ended. Please contact the researcher on Prolific if you need help.",prolific_full:'All study places have been assigned. Please return your submission on Prolific and contact the researcher.',session_required:'This study has already been started. Please use the original browser, or contact the researcher on Prolific for help resuming.',prolific_already_participated:'You have already joined this study. Please use your original session or contact the researcher.',wrong_prolific_study:'This link belongs to a different study. Please return to Prolific and open the correct study link.',study_not_ready:'The study is temporarily unavailable. Please retry shortly or contact the researcher.',consent_version_changed:'The study information has been updated. Please refresh and read the latest version.'};throw new Error(messages[result.error]||'We could not start your study. Please contact the researcher on Prolific.');}
    // After the server session cookie is established, remove IDs from the URL.
    history.replaceState(null,'',reviewDomain?'/review/'+reviewDomain+'/':'/prolific/');
    await onEnrol(result);
   }catch(e){const box=document.getElementById('prolificError');if(box)box.textContent=e.message;}
   finally{enrolling=false;const button=document.getElementById('prolificContinue');if(button)button.disabled=!info.ready||!valid||decision!=='yes';}
  };
 }
 return {load,render,bind};
})();
